#!/usr/bin/env python3
"""
Experiment #009: 1h Volume-Weighted Momentum + 4h HMA Trend Bias + Regime Filter + ATR Stop
Hypothesis: 1h timeframe balances trade frequency with noise reduction. Volume-weighted
momentum captures institutional flow better than price-only signals. 4h HMA provides
HTF trend alignment without excessive lag. Regime filter (ADX + BB Width) adapts
between trend-following (ADX>25) and mean-reversion (ADX<20 + BB squeeze) modes.
Multiple entry paths ensure >=10 trades per symbol. Conservative sizing (0.30) with
2.5*ATR stoploss controls drawdown. Must beat Sharpe=0.499 baseline.
Timeframe: 1h (REQUIRED), HTF: 4h via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_vw_momentum_4h_hma_regime_atr_v1"
timeframe = "1h"
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

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    return upper, lower, sma, bandwidth

def calculate_vw_momentum(close, volume, period=14):
    """
    Volume-Weighted Momentum: measures price change weighted by volume.
    High volume + price up = strong bullish momentum.
    """
    n = len(close)
    vw_mom = np.zeros(n)
    vw_mom[:] = np.nan
    
    for i in range(period, n):
        price_change = close[i] - close[i-period]
        avg_volume = np.mean(volume[i-period+1:i+1])
        vw_mom[i] = price_change * avg_volume / 1e6  # scale down
    
    return vw_mom

def calculate_zscore(series, period=20):
    """Calculate Z-score for mean reversion signals."""
    series_s = pd.Series(series)
    mean = series_s.rolling(window=period, min_periods=period).mean().values
    std = series_s.rolling(window=period, min_periods=period).std().values
    zscore = (series - mean) / np.where(std > 0, std, 1e-10)
    return zscore

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_bw = calculate_bollinger(close, 20, 2.0)
    vw_mom = calculate_vw_momentum(close, volume, 14)
    
    # 1h HMA for trend confirmation
    hma_1h = calculate_hma(close, 21)
    hma_1h_fast = calculate_hma(close, 10)
    
    # Z-score for mean reversion
    close_zscore = calculate_zscore(close, 20)
    
    signals = np.zeros(n)
    SIZE_ENTRY = 0.30
    SIZE_HALF = 0.15
    
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
        
        if np.isnan(adx[i]) or np.isnan(vw_mom[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF)
        htf_bullish = close[i] > hma_4h_aligned[i]
        htf_bearish = close[i] < hma_4h_aligned[i]
        
        # 1h HMA trend
        hma_1h_bullish = close[i] > hma_1h[i]
        hma_1h_bearish = close[i] < hma_1h[i]
        hma_rising = hma_1h[i] > hma_1h[i-1] if i > 0 else False
        hma_falling = hma_1h[i] < hma_1h[i-1] if i > 0 else False
        
        # Fast HMA crossover
        fast_above_slow = hma_1h_fast[i] > hma_1h[i]
        fast_below_slow = hma_1h_fast[i] < hma_1h[i]
        
        # Regime detection
        trend_regime = adx[i] > 25  # Strong trend
        range_regime = adx[i] < 20  # Range-bound
        bb_squeeze = bb_bw[i] < np.nanpercentile(bb_bw[:i], 20) if i > 100 else False
        
        # RSI zones
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_neutral = rsi[i] > 40 and rsi[i] < 60
        
        # Volume-weighted momentum signals
        vw_mom_strong_long = vw_mom[i] > 0 and vw_mom[i] > vw_mom[i-1] if i > 0 else False
        vw_mom_strong_short = vw_mom[i] < 0 and vw_mom[i] < vw_mom[i-1] if i > 0 else False
        
        # Z-score extremes
        zscore_oversold = close_zscore[i] < -1.5
        zscore_overbought = close_zscore[i] > 1.5
        
        # Price vs Bollinger
        near_bb_lower = close[i] < bb_lower[i] * 1.01
        near_bb_upper = close[i] > bb_upper[i] * 0.99
        
        new_signal = 0.0
        
        # === LONG ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: HTF bullish + 1h trend + VW momentum strong + ADX trend regime
        if htf_bullish and hma_1h_bullish and vw_mom_strong_long and trend_regime:
            new_signal = SIZE_ENTRY
        
        # Path 2: HTF bullish + Fast HMA crossover + RSI neutral
        elif htf_bullish and fast_above_slow and rsi_neutral and hma_rising:
            new_signal = SIZE_ENTRY
        
        # Path 3: Range regime + RSI oversold + Near BB lower (mean reversion)
        elif range_regime and rsi_oversold and near_bb_lower:
            new_signal = SIZE_ENTRY
        
        # Path 4: HTF bullish + Z-score oversold (dip buy in uptrend)
        elif htf_bullish and zscore_oversold and rsi[i] > rsi[i-1] if i > 0 else False:
            new_signal = SIZE_ENTRY
        
        # Path 5: BB squeeze breakout + HTF bullish + VW momentum building
        elif bb_squeeze and htf_bullish and vw_mom[i] > 0 and close[i] > bb_sma[i]:
            new_signal = SIZE_ENTRY
        
        # === SHORT ENTRIES (multiple paths for >=10 trades) ===
        
        # Path 1: HTF bearish + 1h trend + VW momentum strong + ADX trend regime
        if htf_bearish and hma_1h_bearish and vw_mom_strong_short and trend_regime:
            new_signal = -SIZE_ENTRY
        
        # Path 2: HTF bearish + Fast HMA crossover down + RSI neutral
        elif htf_bearish and fast_below_slow and rsi_neutral and hma_falling:
            new_signal = -SIZE_ENTRY
        
        # Path 3: Range regime + RSI overbought + Near BB upper (mean reversion)
        elif range_regime and rsi_overbought and near_bb_upper:
            new_signal = -SIZE_ENTRY
        
        # Path 4: HTF bearish + Z-score overbought (rally sell in downtrend)
        elif htf_bearish and zscore_overbought and rsi[i] < rsi[i-1] if i > 0 else False:
            new_signal = -SIZE_ENTRY
        
        # Path 5: BB squeeze breakdown + HTF bearish + VW momentum negative
        elif bb_squeeze and htf_bearish and vw_mom[i] < 0 and close[i] < bb_sma[i]:
            new_signal = -SIZE_ENTRY
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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
            
            # Calculate trailing stop (2.5*ATR for 1h timeframe)
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