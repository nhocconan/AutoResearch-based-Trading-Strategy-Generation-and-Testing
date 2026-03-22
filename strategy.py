#!/usr/bin/env python3
"""
Experiment #011: 12h Hybrid Regime-Adaptive Strategy with 1d/1w HTF
Hypothesis: Combine trend-following (Donchian) in trending regimes with 
mean-reversion (Z-score + BB) in ranging regimes. Use 1d HMA for primary 
trend bias, 1w HMA for macro regime. Bollinger Band Width detects regime:
- BB Width percentile < 20% = squeeze (prepare for breakout)
- BB Width percentile > 80% = extended (mean reversion likely)
ADX confirms trend strength. ATR ratio (7/30) detects vol spikes for 
counter-trend entries. Asymmetric sizing: 0.30 in trends, 0.20 in ranges.
Timeframe: 12h (REQUIRED), HTF: 1d + 1w via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hybrid_regime_1d_1w_hma_bb_zscore_v1"
timeframe = "12h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma
    return upper, lower, sma, width

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion."""
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    zscore = (close - sma) / (std + 1e-10)
    return zscore

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period-1, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    return upper, lower

def calculate_bb_width_percentile(bb_width, lookback=100):
    """Calculate Bollinger Width percentile over lookback period."""
    n = len(bb_width)
    percentile = np.zeros(n)
    percentile[:] = np.nan
    
    for i in range(lookback-1, n):
        if np.all(np.isnan(bb_width[i-lookback+1:i+1])):
            continue
        valid = bb_width[i-lookback+1:i+1]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            percentile[i] = np.sum(bb_width[i] > valid) / len(valid) * 100
    
    return percentile

def calculate_atr_ratio(atr, short_period=7, long_period=30):
    """Calculate ATR ratio for volatility spike detection."""
    atr_s = pd.Series(atr).ewm(span=short_period, min_periods=short_period, adjust=False).mean().values
    atr_l = pd.Series(atr).ewm(span=long_period, min_periods=long_period, adjust=False).mean().values
    ratio = atr_s / (atr_l + 1e-10)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    rsi = calculate_rsi(close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    # Bollinger Bands for regime detection
    bb_upper, bb_lower, bb_mid, bb_width = calculate_bollinger(close, 20, 2.0)
    bb_width_pct = calculate_bb_width_percentile(bb_width, 100)
    
    # Z-score for mean reversion
    zscore = calculate_zscore(close, 20)
    
    # ATR ratio for vol spikes
    atr_ratio = calculate_atr_ratio(atr, 7, 30)
    
    # EMA trend
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    signals = np.zeros(n)
    SIZE_TREND = 0.30
    SIZE_RANGE = 0.20
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(adx[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_width_pct[i]) or np.isnan(zscore[i]) or np.isnan(atr_ratio[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME DETECTION ===
        # Low BB width percentile = squeeze (trend breakout likely)
        # High BB width percentile = extended (mean reversion likely)
        regime_squeeze = bb_width_pct[i] < 25
        regime_extended = bb_width_pct[i] > 75
        
        # === HTF TREND BIAS ===
        hma_1d_bullish = close[i] > hma_1d_aligned[i]
        hma_1d_bearish = close[i] < hma_1d_aligned[i]
        hma_1w_bullish = close[i] > hma_1w_aligned[i]
        hma_1w_bearish = close[i] < hma_1w_aligned[i]
        
        # === LOCAL TREND ===
        ema_bullish = close[i] > ema_21[i] and ema_21[i] > ema_50[i]
        ema_bearish = close[i] < ema_21[i] and ema_21[i] < ema_50[i]
        
        # === ADX TREND STRENGTH ===
        trend_strong = adx[i] > 22
        trend_weak = adx[i] < 18
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 1.8
        
        # === DONCHIAN BREAKOUT ===
        breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        new_signal = 0.0
        current_size = SIZE_TREND if trend_strong else SIZE_RANGE
        
        # === TREND-FOLLOWING ENTRIES (squeeze regime + breakout) ===
        if regime_squeeze and trend_strong:
            # Long breakout with HTF and local trend alignment
            if breakout_long and hma_1d_bullish and ema_bullish and rsi[i] < 70:
                new_signal = current_size
            # Short breakout with HTF and local trend alignment
            elif breakout_short and hma_1d_bearish and ema_bearish and rsi[i] > 30:
                new_signal = -current_size
        
        # === MEAN REVERSION ENTRIES (extended regime + z-score) ===
        if regime_extended or trend_weak:
            # Long when deeply oversold + vol spike + HTF not strongly bearish
            if zscore[i] < -2.0 and vol_spike and not hma_1w_bearish:
                new_signal = SIZE_RANGE
            # Short when deeply overbought + vol spike + HTF not strongly bullish
            elif zscore[i] > 2.0 and vol_spike and not hma_1w_bullish:
                new_signal = -SIZE_RANGE
            
            # BB touch entries
            if close[i] <= bb_lower[i] and rsi[i] < 35 and not hma_1w_bearish:
                new_signal = SIZE_RANGE
            elif close[i] >= bb_upper[i] and rsi[i] > 65 and not hma_1w_bullish:
                new_signal = -SIZE_RANGE
        
        # === TREND CONTINUATION (pullback to EMA in strong trend) ===
        if trend_strong and hma_1d_bullish and hma_1w_bullish:
            if ema_bullish and rsi[i] > 45 and rsi[i] < 60 and close[i] > ema_21[i] * 0.98:
                new_signal = SIZE_TREND
        
        if trend_strong and hma_1d_bearish and hma_1w_bearish:
            if ema_bearish and rsi[i] > 40 and rsi[i] < 55 and close[i] < ema_21[i] * 1.02:
                new_signal = -SIZE_TREND
        
        # === STOPLOSS LOGIC ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = highest_close - 2.5 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (2.5*ATR)
            current_stop = lowest_close + 2.5 * atr[i]
            if trailing_stop == 0.0 or current_stop < trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] > trailing_stop:
                new_signal = 0.0
        
        # Update position tracking AFTER signal calculation
        prev_signal = signals[i - 1] if i > 0 else 0.0
        
        # New position opened
        if new_signal != 0.0 and prev_signal == 0.0:
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 2.5 * atr[i] if position_side > 0 else close[i] + 2.5 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position closed
        elif new_signal == 0.0 and prev_signal != 0.0:
            position_side = 0
            entry_price = 0.0
            trailing_stop = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals