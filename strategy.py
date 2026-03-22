#!/usr/bin/env python3
"""
Experiment #001: 15m Multi-Regime Strategy with 4h HMA Bias + RSI Mean Reversion + Choppiness Filter
Hypothesis: 15m timeframe captures intraday moves while 4h HTF provides trend bias.
Combines regime detection (Choppiness Index) to switch between trend-following and mean-reversion.
RSI extremes + Bollinger Bands for mean-reversion entries in choppy markets.
Supertrend + HMA alignment for trend entries in trending markets.
Multiple entry paths ensure >=10 trades per symbol. Conservative sizing (0.25) controls DD.
2.5*ATR stoploss for 15m bars. Must work on BTC/ETH/SOL individually (Sharpe > 0 each).
Timeframe: 15m (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_regime_rsi_chop_4h_hma_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    return upper, lower, sma, width

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = range/choppy (mean reversion works)
    CHOP < 38.2 = trending (trend following works)
    """
    n = len(close)
    chop = np.zeros(n)
    chop[:] = np.nan
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
            continue
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    return chop

def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Supertrend indicator - trend following with ATR bands."""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.zeros(n)
    direction[:] = np.nan
    
    hl2 = (high + low) / 2.0
    
    for i in range(period, n):
        if np.isnan(atr[i]) or atr[i] == 0:
            supertrend[i] = np.nan
            direction[i] = np.nan
            continue
        
        upper_band = hl2[i] + multiplier * atr[i]
        lower_band = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band
            direction[i] = 1
        else:
            if close[i] > supertrend[i-1]:
                supertrend[i] = lower_band
                direction[i] = 1
            elif close[i] < supertrend[i-1]:
                supertrend[i] = upper_band
                direction[i] = -1
            else:
                if direction[i-1] == 1:
                    supertrend[i] = max(lower_band, supertrend[i-1])
                    direction[i] = 1
                else:
                    supertrend[i] = min(upper_band, supertrend[i-1])
                    direction[i] = -1
    
    return supertrend, direction

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI) - combines RSI, streak RSI, and percentile rank.
    CRSI = (RSI(close,3) + RSI(streak,2) + PercentRank(100)) / 3
    Long: CRSI < 10, Short: CRSI > 90
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if i > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if i > 0 else -1
        else:
            streak[i] = 0
    
    streak_abs = np.abs(streak)
    streak_rsi = calculate_rsi(streak_abs + 1, streak_period)  # +1 to avoid zeros
    
    # Percentile Rank
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        rank = np.sum(window[:-1] < close[i]) / (rank_period - 1) * 100
        
        crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 15m indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    rsi_fast = calculate_rsi(close, 7)
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    chop = calculate_choppiness_index(high, low, close, 14)
    supertrend, st_direction = calculate_supertrend(high, low, close, 10, 3.0)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # 15m HMA for trend
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.25
    SIZE_HALF = 0.12
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]) or np.isnan(bb_width[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m regime detection
        is_choppy = chop[i] > 55  # Range market - mean reversion
        is_trending = chop[i] < 45  # Trending market - trend following
        
        # 15m Supertrend
        st_bullish = st_direction[i] == 1
        st_bearish = st_direction[i] == -1
        st_flip_long = st_direction[i] == 1 and st_direction[i-1] == -1 if i > 0 else False
        st_flip_short = st_direction[i] == -1 and st_direction[i-1] == 1 if i > 0 else False
        
        # 15m HMA
        hma_bullish = close[i] > hma_15m[i]
        hma_bearish = close[i] < hma_15m[i]
        hma_rising = hma_15m[i] > hma_15m[i-1] if i > 0 else False
        hma_falling = hma_15m[i] < hma_15m[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_15m_fast[i] > hma_15m[i]
        fast_below_slow = hma_15m_fast[i] < hma_15m[i]
        
        # RSI zones
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        rsi_extreme_oversold = rsi[i] < 20
        rsi_extreme_overbought = rsi[i] > 80
        
        # Bollinger Bands
        price_at_lower = close[i] <= bb_lower[i]
        price_at_upper = close[i] >= bb_upper[i]
        bb_squeeze = bb_width[i] < np.nanmean(bb_width[max(0,i-100):i]) * 0.7 if i > 100 else False
        
        # Connors RSI
        crsi_oversold = crsi[i] < 15 if not np.isnan(crsi[i]) else False
        crsi_overbought = crsi[i] > 85 if not np.isnan(crsi[i]) else False
        
        new_signal = 0.0
        
        # === MEAN REVERSION ENTRIES (choppy market) ===
        
        # Path 1: RSI oversold + price at BB lower + choppy market
        if is_choppy and rsi_oversold and price_at_lower and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: RSI overbought + price at BB upper + choppy market
        elif is_choppy and rsi_overbought and price_at_upper and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Connors RSI extreme oversold + HTF bullish
        elif crsi_oversold and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 4: Connors RSI extreme overbought + HTF bearish
        elif crsi_overbought and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 5: BB squeeze breakout long
        elif bb_squeeze and close[i] > bb_upper[i] and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 6: BB squeeze breakout short
        elif bb_squeeze and close[i] < bb_lower[i] and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # === TREND FOLLOWING ENTRIES (trending market) ===
        
        # Path 7: Supertrend flip long + HTF bullish + trending
        if is_trending and st_flip_long and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 8: Supertrend flip short + HTF bearish + trending
        elif is_trending and st_flip_short and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 9: HMA crossover up + HTF bullish + trending
        elif is_trending and fast_above_slow and htf_bullish and hma_rising:
            new_signal = SIZE_ENTRY
        
        # Path 10: HMA crossover down + HTF bearish + trending
        elif is_trending and fast_below_slow and htf_bearish and hma_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 11: Supertrend bullish + RSI pullback + HTF bullish
        elif st_bullish and rsi[i] > 40 and rsi[i] < 55 and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 12: Supertrend bearish + RSI pullback + HTF bearish
        elif st_bearish and rsi[i] > 45 and rsi[i] < 60 and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m timeframe)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 15m timeframe)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.5 * atr[i]
                profit = entry_price - close[i]
                if profit >= 2.0 * risk:
                    new_signal = -SIZE_HALF
                    position_reduced = True
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reduced (take profit)
        elif new_signal != 0.0 and prev_signal != 0.0 and np.abs(new_signal) < np.abs(prev_signal):
            position_reduced = True
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
            position_reduced = False
        
        signals[i] = new_signal
    
    return signals