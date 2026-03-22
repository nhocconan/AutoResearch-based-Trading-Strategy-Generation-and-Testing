#!/usr/bin/env python3
"""
Experiment #020: 30m Volatility Squeeze Breakout with 4h HMA Regime Filter
Hypothesis: Bollinger Band squeeze (low volatility) followed by expansion captures major moves.
4h HMA provides trend bias to filter counter-trend breakouts. Fisher Transform for precise entry timing.
ATR ratio detects vol spikes (panic bottoms/tops) for mean reversion entries.
Asymmetric sizing: 0.25 base, reduce to 0.15 in choppy regimes (ADX<20).
This differs from failed 30m strategies by: (1) waiting for volatility contraction first,
(2) requiring volume confirmation on breakout, (3) wider 3*ATR stops to avoid whipsaw.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25 max, discrete levels (0.0, ±0.15, ±0.25).
Stoploss: 3.0*ATR trailing stop (wider than previous to reduce premature exits).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_bb_squeeze_fisher_4h_hma_vol_v1"
timeframe = "30m"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and bandwidth."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    bandwidth = (upper - lower) / sma
    bandwidth[np.isnan(bandwidth)] = 0.0
    bandwidth[np.isinf(bandwidth)] = 0.0
    return upper, lower, sma, bandwidth

def calculate_fisher(close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher[:] = np.nan
    fisher_signal = np.zeros(n)
    fisher_signal[:] = np.nan
    
    for i in range(period - 1, n):
        hh = np.max(close[i - period + 1:i + 1])
        ll = np.min(close[i - period + 1:i + 1])
        
        if hh > ll:
            normalized = 0.66 * ((close[i] - ll) / (hh - ll) - 0.5)
            if i > period - 1:
                normalized += 0.67 * fisher_signal[i - 1]
            normalized = np.clip(normalized, -0.99, 0.99)
            fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
            if i > period:
                fisher_signal[i] = fisher[i - 1]
            else:
                fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / atr[mask]
    
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_ma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    ratio = volume / vol_ma
    ratio[np.isnan(ratio)] = 1.0
    ratio[np.isinf(ratio)] = 1.0
    return ratio

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio for volatility spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    ratio = atr_short / atr_long
    ratio[np.isnan(ratio)] = 1.0
    ratio[np.isinf(ratio)] = 1.0
    return ratio

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_sma, bb_bandwidth = calculate_bollinger_bands(close, 20, 2.0)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    adx = calculate_adx(high, low, close, 14)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate BB bandwidth percentile for squeeze detection
    bb_bw_percentile = pd.Series(bb_bandwidth).rolling(window=100, min_periods=50).apply(
        lambda x: np.percentile(x[~np.isnan(x)], 20) if len(x[~np.isnan(x)]) > 0 else np.nan
    ).values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels
    SIZE_BASE = 0.25
    SIZE_REDUCED = 0.15  # In choppy/low ADX regimes
    SIZE_EXIT = 0.0
    
    # Track positions for stoploss
    position_side = 0
    entry_price = 0.0
    trailing_stop = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(bb_bandwidth[i]):
            signals[i] = 0.0
            continue
        
        # 4h regime bias (HTF) - determines which direction to favor
        bull_regime = close[i] > hma_4h_aligned[i]
        bear_regime = close[i] < hma_4h_aligned[i]
        
        # ADX regime - trending vs choppy
        trending = adx[i] > 25
        choppy = adx[i] < 20
        
        # BB squeeze detection (bandwidth at low percentile)
        bb_squeeze = bb_bandwidth[i] < np.nanpercentile(bb_bandwidth[max(0,i-100):i+1], 25)
        
        # BB expansion (breakout from squeeze)
        bb_expanding = bb_bandwidth[i] > bb_bandwidth[i-5] if i > 5 else False
        
        # Price position vs BB
        price_near_lower = close[i] < bb_lower[i] * 1.01
        price_near_upper = close[i] > bb_upper[i] * 0.99
        price_above_sma = close[i] > bb_sma[i]
        price_below_sma = close[i] < bb_sma[i]
        
        # Fisher Transform signals
        fisher_long_cross = fisher[i] > -1.5 and fisher_signal[i] < -1.5 if i > 0 else False
        fisher_short_cross = fisher[i] < 1.5 and fisher_signal[i] > 1.5 if i > 0 else False
        
        # Fisher extreme levels (mean reversion)
        fisher_oversold = fisher[i] < -2.0
        fisher_overbought = fisher[i] > 2.0
        
        # Volume confirmation
        volume_confirmed = vol_ratio[i] > 1.3
        
        # Vol spike detection (panic conditions)
        vol_spike = atr_ratio[i] > 2.0
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and ema_50[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and ema_50[i] < ema_200[i]
        
        # Select position size based on regime
        current_size = SIZE_REDUCED if choppy else SIZE_BASE
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: BB squeeze + expansion + bull regime + volume + Fisher long
        if bb_squeeze and bb_expanding and bull_regime and volume_confirmed and fisher_long_cross:
            new_signal = current_size
        # Secondary: Vol spike + price near lower BB + Fisher oversold (panic bottom)
        elif vol_spike and price_near_lower and fisher_oversold:
            new_signal = current_size
        # Tertiary: EMA bullish + price above BB SMA + bull regime + trending
        elif ema_bullish and price_above_sma and bull_regime and trending:
            new_signal = current_size
        # Quaternary: Price breaks above BB upper + volume + bull regime
        elif price_near_upper and volume_confirmed and bull_regime and bb_expanding:
            new_signal = current_size
        
        # === SHORT ENTRY ===
        # Primary: BB squeeze + expansion + bear regime + volume + Fisher short
        if bb_squeeze and bb_expanding and bear_regime and volume_confirmed and fisher_short_cross:
            new_signal = -current_size
        # Secondary: Vol spike + price near upper BB + Fisher overbought (panic top)
        elif vol_spike and price_near_upper and fisher_overbought:
            new_signal = -current_size
        # Tertiary: EMA bearish + price below BB SMA + bear regime + trending
        elif ema_bearish and price_below_sma and bear_regime and trending:
            new_signal = -current_size
        # Quaternary: Price breaks below BB lower + volume + bear regime
        elif price_near_lower and volume_confirmed and bear_regime and bb_expanding:
            new_signal = -current_size
        
        # === STOPLOSS LOGIC (Rule 6) ===
        if position_side > 0 and entry_price > 0:
            # Update highest close for trailing
            if close[i] > highest_close:
                highest_close = close[i]
            
            # Calculate trailing stop (3.0*ATR - wider to avoid whipsaw)
            current_stop = highest_close - 3.0 * atr[i]
            if current_stop > trailing_stop:
                trailing_stop = current_stop
            
            # Check stoploss hit
            if close[i] < trailing_stop:
                new_signal = 0.0
        
        if position_side < 0 and entry_price > 0:
            # Update lowest close for trailing
            if close[i] < lowest_close or lowest_close == 0.0:
                lowest_close = close[i]
            
            # Calculate trailing stop (3.0*ATR)
            current_stop = lowest_close + 3.0 * atr[i]
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
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        
        # Position reversed
        elif new_signal != 0.0 and prev_signal != 0.0 and np.sign(new_signal) != np.sign(prev_signal):
            entry_price = close[i]
            position_side = np.sign(new_signal)
            trailing_stop = close[i] - 3.0 * atr[i] if position_side > 0 else close[i] + 3.0 * atr[i]
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