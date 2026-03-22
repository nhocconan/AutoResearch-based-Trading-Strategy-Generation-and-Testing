#!/usr/bin/env python3
"""
Experiment #019: 15m Connors RSI Mean Reversion with 4h HTF Trend Filter
Hypothesis: 15m timeframe captures intraday mean reversion opportunities while
4h HMA provides trend bias. Connors RSI (CRSI) has proven 75% win rate for
reversals. Combined with Bollinger Band squeeze detection and volume confirmation,
this should work in both bull and bear markets. Conservative sizing (0.25-0.30)
with 2.0*ATR stoploss. Multiple entry paths ensure >=10 trades per symbol.
Timeframe: 15m (REQUIRED), HTF: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_15m_crsi_4h_hma_bb_squeeze_vol_atr_v1"
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

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(100)) / 3
    Proven 75% win rate for mean reversion entries.
    """
    n = len(close)
    crsi = np.zeros(n)
    crsi[:] = np.nan
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Percent Rank component
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window[:-1] < current) / (rank_period - 1) * 100
        crsi[i] = (rsi_short[i] + streak_rsi[i] + rank) / 3.0
    
    return crsi

def calculate_zscore(close, period=20):
    """Calculate Z-score of price relative to rolling mean."""
    close_s = pd.Series(close)
    mean = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    zscore = (close - mean) / (std + 1e-10)
    return zscore

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    mid = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    bw = (upper - lower) / mid  # Bandwidth
    return upper, mid, lower, bw

def calculate_volume_ratio(taker_buy_volume, volume):
    """Calculate taker buy volume ratio (0-1, >0.5 = bullish pressure)."""
    ratio = np.zeros(len(volume))
    mask = volume > 0
    ratio[mask] = taker_buy_volume[mask] / volume[mask]
    return ratio

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    atr = calculate_atr(high, low, close, period)
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    for i in range(period, n):
        if atr[i] > 0:
            plus_di[i] = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
            minus_di[i] = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values[i] / atr[i]
    
    dx = np.zeros(n)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / di_sum
    
    adx[period:] = pd.Series(dx[period:]).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

def calculate_volatility_spike(atr, period_short=7, period_long=30):
    """Calculate ATR ratio for volatility spike detection."""
    atr_s = pd.Series(atr)
    atr_short = atr_s.ewm(span=period_short, min_periods=period_short, adjust=False).mean().values
    atr_long = atr_s.ewm(span=period_long, min_periods=period_long, adjust=False).mean().values
    ratio = atr_short / (atr_long + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
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
    crsi = calculate_crsi(close, 3, 2, 100)
    zscore = calculate_zscore(close, 20)
    adx = calculate_adx(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(taker_buy_vol, volume)
    vol_spike = calculate_volatility_spike(atr, 7, 30)
    
    # Bollinger Bands
    bb_upper, bb_mid, bb_lower, bb_bw = calculate_bollinger(close, 20, 2.0)
    
    # SMA 200 for long-term trend
    sma_200 = calculate_sma(close, 200)
    
    # HMA for 15m trend
    hma_15m = calculate_hma(close, 21)
    hma_15m_fast = calculate_hma(close, 10)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.28
    SIZE_HALF = 0.14
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    position_reduced = False
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(adx[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(crsi[i]) or np.isnan(vol_spike[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 15m trend
        hma_15m_bullish = close[i] > hma_15m[i]
        hma_15m_bearish = close[i] < hma_15m[i]
        
        # HMA crossover
        fast_above_slow = hma_15m_fast[i] > hma_15m[i]
        fast_below_slow = hma_15m_fast[i] < hma_15m[i]
        
        # Volume confirmation
        vol_bullish = vol_ratio[i] > 0.48
        vol_bearish = vol_ratio[i] < 0.52
        
        # ADX regime
        trend_strong = adx[i] > 20
        trend_weak = adx[i] < 25
        
        # Z-score extremes
        zscore_oversold = zscore[i] < -1.5
        zscore_overbought = zscore[i] > 1.5
        
        # RSI extremes
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_fast_oversold = rsi_fast[i] < 25
        rsi_fast_overbought = rsi_fast[i] > 75
        
        # CRSI extremes (Connors RSI)
        crsi_oversold = crsi[i] < 10
        crsi_overbought = crsi[i] > 90
        
        # Bollinger Band position
        price_near_lower = close[i] < bb_lower[i] * 1.005
        price_near_upper = close[i] > bb_upper[i] * 0.995
        bb_squeeze = bb_bw[i] < np.nanpercentile(bb_bw[:i], 20) if i > 20 else False
        
        # Volatility spike
        vol_spike_high = vol_spike[i] > 2.0
        
        # SMA 200 filter
        price_above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        price_below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        new_signal = 0.0
        
        # === LONG ENTRIES (Mean Reversion Focus) ===
        
        # Path 1: CRSI oversold + 4h bullish (primary mean reversion)
        if crsi_oversold and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 2: CRSI oversold + price at BB lower + volume bullish (deep pullback)
        elif crsi_oversold and price_near_lower and vol_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 3: Z-score oversold + 4h bullish + ADX weak (mean reversion in ranging uptrend)
        elif zscore_oversold and htf_bullish and trend_weak:
            new_signal = SIZE_ENTRY
        
        # Path 4: RSI fast oversold + CRSI oversold (double confirmation)
        elif rsi_fast_oversold and crsi_oversold:
            new_signal = SIZE_ENTRY
        
        # Path 5: Volatility spike + price at BB lower + 4h bullish (vol crush play)
        elif vol_spike_high and price_near_lower and htf_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 6: BB squeeze breakout long + 4h bullish + volume
        elif bb_squeeze and close[i] > bb_upper[i] and htf_bullish and vol_bullish:
            new_signal = SIZE_ENTRY
        
        # Path 7: HMA crossover up + 4h bullish + RSI not overbought
        elif fast_above_slow and htf_bullish and rsi[i] < 70:
            new_signal = SIZE_ENTRY
        
        # Path 8: Price above SMA200 + CRSI oversold (bull market dip buy)
        elif price_above_sma200 and crsi_oversold:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (Mean Reversion Focus) ===
        
        # Path 1: CRSI overbought + 4h bearish (primary mean reversion)
        if crsi_overbought and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 2: CRSI overbought + price at BB upper + volume bearish
        elif crsi_overbought and price_near_upper and vol_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Z-score overbought + 4h bearish + ADX weak
        elif zscore_overbought and htf_bearish and trend_weak:
            new_signal = -SIZE_ENTRY
        
        # Path 4: RSI fast overbought + CRSI overbought (double confirmation)
        elif rsi_fast_overbought and crsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # Path 5: Volatility spike + price at BB upper + 4h bearish
        elif vol_spike_high and price_near_upper and htf_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 6: BB squeeze breakout short + 4h bearish + volume
        elif bb_squeeze and close[i] < bb_lower[i] and htf_bearish and vol_bearish:
            new_signal = -SIZE_ENTRY
        
        # Path 7: HMA crossover down + 4h bearish + RSI not oversold
        elif fast_below_slow and htf_bearish and rsi[i] > 30:
            new_signal = -SIZE_ENTRY
        
        # Path 8: Price below SMA200 + CRSI overbought (bear market rally sell)
        elif price_below_sma200 and crsi_overbought:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
            current_stop = highest_close - 2.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
                profit = close[i] - entry_price
                if profit >= 2.0 * risk:
                    new_signal = SIZE_HALF
                    position_reduced = True
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.0*ATR for 15m timeframe)
            current_stop = lowest_close + 2.0 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
            elif not position_reduced:
                # Take profit at 2R
                risk = 2.0 * atr[i]
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
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
            position_reduced = False
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.0 * atr[i] if position_side > 0 else close[i] + 2.0 * atr[i]
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