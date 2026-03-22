#!/usr/bin/env python3
"""
Experiment #011: 12h Asymmetric Regime + 1d HMA Trend + Volume Confirmation
Hypothesis: BTC/ETH perform differently in bull vs bear regimes. Use 1d HMA to determine
primary trend bias, then apply ASYMMETRIC logic:
- BULL regime (price > 1d HMA): Only take long breakouts, avoid shorts except extreme overbought
- BEAR regime (price < 1d HMA): Only take short breakouts, avoid longs except extreme oversold
- RANGE regime (ADX < 20): Mean reversion at Bollinger bands regardless of trend

Key innovations:
1. Asymmetric entry logic (different rules for bull vs bear) - avoids counter-trend failures
2. Volume confirmation on breakouts (volume > 1.5x 20-bar avg) - filters false breakouts
3. 12h timeframe captures multi-day swings without 4h noise or 1d lag
4. 1d HMA via mtf_data helper for proper HTF alignment (no look-ahead)
5. Conservative sizing (0.25 base, 0.35 max) with 2.5*ATR trailing stop

Timeframe: 12h (REQUIRED), HTF: 1d via mtf_data helper.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_asymmetric_1d_hma_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    
    mask = tr_smooth > 0
    di_plus[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    di_minus[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    mask2 = (di_plus + di_minus) > 0
    dx[mask2] = 100 * np.abs(di_plus[mask2] - di_minus[mask2]) / (di_plus[mask2] + di_minus[mask2])
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx[period * 2:] = adx_raw[period * 2:]
    
    return adx

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    return upper, lower, bandwidth, sma

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    n = len(close)
    rsi = np.zeros(n)
    rsi[:] = np.nan
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gains).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(losses).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = avg_loss > 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[~mask] = 100.0
    
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high / lowest low over period)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    upper[:] = np.nan
    lower[:] = np.nan
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.zeros(n)
    kama[:] = np.nan
    
    # Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1))[1:])
    
    er = np.zeros(n)
    mask = volatility > 0
    er[mask] = change[mask] / volatility[mask]
    er[:er_period] = 0.0
    
    # Smoothing Constant
    sc = (er * (2.0 / (fast_period + 1) - 2.0 / (slow_period + 1)) + 2.0 / (slow_period + 1)) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    bb_upper, bb_lower, bb_bandwidth, bb_sma = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    kama = calculate_kama(close, 10, 2, 30)
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.35
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(donchian_upper[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        # 1d trend bias (HTF) - determines regime
        bull_regime = close[i] > hma_1d_aligned[i]
        bear_regime = close[i] < hma_1d_aligned[i]
        
        # Trend strength
        trending = adx[i] > 25
        ranging = adx[i] < 20
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i - 1] if not np.isnan(donchian_upper[i - 1]) else False
        breakout_short = close[i] < donchian_lower[i - 1] if not np.isnan(donchian_lower[i - 1]) else False
        
        # Bollinger mean reversion signals
        price_below_lower = close[i] < bb_lower[i] * 1.005
        price_above_upper = close[i] > bb_upper[i] * 0.995
        
        # RSI extremes
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        rsi_extreme_oversold = rsi[i] < 25
        rsi_extreme_overbought = rsi[i] > 75
        
        # KAMA trend
        kama_bullish = close[i] > kama[i]
        kama_bearish = close[i] < kama[i]
        
        new_signal = 0.0
        
        # === BULL REGIME (price > 1d HMA): Favor longs, avoid shorts ===
        if bull_regime:
            # Strong long: Donchian breakout + volume + KAMA confirmation
            if breakout_long and volume_confirmed and kama_bullish:
                new_signal = SIZE_MAX
            # Moderate long: Donchian breakout in bull regime
            elif breakout_long and bull_regime:
                new_signal = SIZE_BASE
            # Mean reversion long: Only at extreme oversold in bull regime
            elif price_below_lower and rsi_extreme_oversold:
                new_signal = SIZE_BASE
            # Avoid shorts except extreme overbought
            elif price_above_upper and rsi_extreme_overbought and ranging:
                new_signal = -SIZE_BASE
        
        # === BEAR REGIME (price < 1d HMA): Favor shorts, avoid longs ===
        elif bear_regime:
            # Strong short: Donchian breakout + volume + KAMA confirmation
            if breakout_short and volume_confirmed and kama_bearish:
                new_signal = -SIZE_MAX
            # Moderate short: Donchian breakout in bear regime
            elif breakout_short and bear_regime:
                new_signal = -SIZE_BASE
            # Mean reversion short: Only at extreme overbought in bear regime
            elif price_above_upper and rsi_extreme_overbought:
                new_signal = -SIZE_BASE
            # Avoid longs except extreme oversold
            elif price_below_lower and rsi_extreme_oversold and ranging:
                new_signal = SIZE_BASE
        
        # === RANGE REGIME (ADX < 20): Pure mean reversion ===
        if ranging:
            # Long at lower BB with RSI oversold
            if price_below_lower and rsi_oversold:
                new_signal = SIZE_BASE
            # Short at upper BB with RSI overbought
            elif price_above_upper and rsi_overbought:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) ===
        # Long position stoploss
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
        
        # Short position stoploss
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