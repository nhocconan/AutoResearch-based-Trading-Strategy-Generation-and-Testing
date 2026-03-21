#!/usr/bin/env python3
"""
Experiment #001: 15m Multi-Timeframe Mean Reversion + Trend Filter
Hypothesis: 15m timeframe captures intraday swings while 4h HMA filters major trend.
Uses Connors RSI (CRSI) for mean reversion entries in trend direction.
Choppiness Index detects regime: range (mean revert) vs trend (trend follow).
4h HMA provides major trend bias - only long when 4h bullish, short when bearish.
ATR stoploss at 2.5x protects against 2022-style crashes.
Position sizing: 0.25 base, 0.30 on strong signals, discrete levels to reduce churn.
This should generate 50-100 trades/year with better risk-adjusted returns than daily.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_crsi_chop_15m_v1"
timeframe = "15m"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI indicator."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    rs = np.where(avg_l > 0, avg_g / avg_l, 100.0)
    rsi = 100 - 100 / (1 + rs)
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Extreme readings (<10 or >90) signal mean reversion opportunities.
    """
    n = len(close)
    
    # RSI(3) - fast RSI for short-term extremes
    rsi_fast = calculate_rsi(close, rsi_period)
    
    # RSI Streak - RSI of consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_abs = np.abs(streak)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] > 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        elif streak[i] < 0:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current price ranks in last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < window[-1]) / (rank_period - 1) * 100
        percent_rank[i] = rank
    
    crsi = (rsi_fast + streak_rsi + percent_rank) / 3
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP) measures market choppy vs trending.
    CHOP > 61.8 = range-bound (mean reversion favorable)
    CHOP < 38.2 = trending (trend following favorable)
    Formula: 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        atr_sum = 0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for trend direction."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return wma3.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    # Bandwidth for regime detection
    bw = (upper - lower) / sma * 100
    return upper, lower, sma, bw

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load 4h HTF data ONCE before loop (Rule 1)
    df_4h = get_htf_data(prices, '4h')
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)  # auto shift(1)
    
    # Also load 1h for intermediate trend
    df_1h = get_htf_data(prices, '1h')
    hma_1h = calculate_hma(df_1h['close'].values, 21)
    hma_1h_aligned = align_htf_to_ltf(prices, df_1h, hma_1h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_crsi(close, 3, 2, 100)
    chop = calculate_choppiness(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid, bb_bw = calculate_bollinger(close, 20, 2.0)
    
    # Volume SMA for confirmation
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_sma = np.nan_to_num(vol_sma, nan=0.0)
    
    # Bollinger Band Width percentile for regime
    bb_bw_percentile = pd.Series(bb_bw).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x[x>0], 50) if len(x[x>0]) > 0 else 50, raw=False
    ).values
    bb_bw_percentile = np.nan_to_num(bb_bw_percentile, nan=50)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    HALF_SIZE = 0.12
    
    # Track positions for stoploss
    entry_price = np.zeros(n)
    position_side = 0
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(150, n):
        # 4h major trend filter
        hma_4h_val = hma_4h_aligned[i]
        hma_4h_prev = hma_4h_aligned[i-1] if i > 0 else hma_4h_val
        trend_4h_bullish = hma_4h_val > 0 and close[i] > hma_4h_val
        trend_4h_bearish = hma_4h_val > 0 and close[i] < hma_4h_val
        
        # 1h intermediate trend
        hma_1h_val = hma_1h_aligned[i]
        trend_1h_bullish = hma_1h_val > 0 and close[i] > hma_1h_val
        trend_1h_bearish = hma_1h_val > 0 and close[i] < hma_1h_val
        
        # Regime detection
        is_choppy = chop[i] > 55  # Range-bound market
        is_trending = chop[i] < 45  # Trending market
        
        # Connors RSI extremes for mean reversion
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        crsi_neutral = crsi[i] > 35 and crsi[i] < 65
        
        # Bollinger position
        price_vs_bb_long = close[i] < bb_lower[i] * 1.005  # Near or below lower band
        price_vs_bb_short = close[i] > bb_upper[i] * 0.995  # Near or above upper band
        
        # RSI confirmation
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # Volume confirmation
        vol_confirm = volume[i] > vol_sma[i] * 0.7 if vol_sma[i] > 0 else True
        
        new_signal = 0.0
        
        # LONG ENTRY: 4h bullish + mean reversion signal
        if trend_4h_bullish:
            # Strong long: CRSI oversold + BB lower + volume
            if crsi_oversold and price_vs_bb_long and vol_confirm:
                new_signal = STRONG_SIZE
            # Moderate long: RSI oversold + 1h bullish
            elif rsi_oversold and trend_1h_bullish and crsi[i] < 30:
                new_signal = BASE_SIZE
            # Trend continuation: CRSI neutral + price above BB mid
            elif crsi_neutral and close[i] > bb_mid[i] and trend_1h_bullish:
                new_signal = BASE_SIZE
        
        # SHORT ENTRY: 4h bearish + mean reversion signal
        elif trend_4h_bearish:
            # Strong short: CRSI overbought + BB upper + volume
            if crsi_overbought and price_vs_bb_short and vol_confirm:
                new_signal = -STRONG_SIZE
            # Moderate short: RSI overbought + 1h bearish
            elif rsi_overbought and trend_1h_bearish and crsi[i] > 70:
                new_signal = -BASE_SIZE
            # Trend continuation: CRSI neutral + price below BB mid
            elif crsi_neutral and close[i] < bb_mid[i] and trend_1h_bearish:
                new_signal = -BASE_SIZE
        
        # Choppy market: tighter mean reversion (both directions allowed)
        if is_choppy and not trend_4h_bullish and not trend_4h_bearish:
            if crsi_oversold and price_vs_bb_long:
                new_signal = BASE_SIZE
            elif crsi_overbought and price_vs_bb_short:
                new_signal = -BASE_SIZE
        
        # Stoploss logic (Rule 6) - ATR based
        if position_side > 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] - 2.5 * atr[i]
            if close[i] < stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for longs - take partial profit at 3R
            elif close[i] > entry_price[i-1] + 3.0 * atr[i]:
                if signals[i-1] > 0 and new_signal == 0:
                    new_signal = HALF_SIZE
        
        if position_side < 0 and entry_price[i-1] > 0:
            stop_loss = entry_price[i-1] + 2.5 * atr[i]
            if close[i] > stop_loss:
                new_signal = 0.0  # Stoploss hit
            # Trail stop for shorts - take partial profit at 3R
            elif close[i] < entry_price[i-1] - 3.0 * atr[i]:
                if signals[i-1] < 0 and new_signal == 0:
                    new_signal = -HALF_SIZE
        
        # Update position tracking
        if new_signal != 0 and position_side == 0:
            entry_price[i] = close[i]
            position_side = np.sign(new_signal)
            highest_since_entry[i] = close[i]
            lowest_since_entry[i] = close[i]
        elif new_signal != 0 and position_side != 0:
            if np.sign(new_signal) != position_side:
                entry_price[i] = close[i]
                position_side = np.sign(new_signal)
                highest_since_entry[i] = close[i]
                lowest_since_entry[i] = close[i]
            else:
                entry_price[i] = entry_price[i-1]
                highest_since_entry[i] = max(highest_since_entry[i-1], close[i])
                lowest_since_entry[i] = min(lowest_since_entry[i-1], close[i])
        else:
            entry_price[i] = entry_price[i-1] if i > 0 else 0
            highest_since_entry[i] = highest_since_entry[i-1] if i > 0 else close[i]
            lowest_since_entry[i] = lowest_since_entry[i-1] if i > 0 else close[i]
            if position_side != 0 and new_signal == 0:
                position_side = 0  # Position closed
        
        signals[i] = new_signal
    
    return signals