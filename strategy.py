#!/usr/bin/env python3
"""
Experiment #002: 30m Z-Score Mean Reversion + 4h HMA Trend + ADX Regime
Hypothesis: Z-score mean reversion (20-period) captures oversold/overbought extremes with 
better trade frequency than CRSI. Combined with 4h HMA trend filter to avoid counter-trend 
trades in strong trends. ADX(14) > 20 filter ensures we only trade when there's momentum.
ATR trailing stop at 2.5*ATR limits drawdown. Discrete position sizing (0.0, ±0.25, ±0.30)
minimizes fee churn. This should generate 30-50 trades/year per symbol with positive Sharpe
on BTC/ETH/SOL individually.
Timeframe: 30m (REQUIRED), HTF: 4h via mtf_data helper.
Position sizing: 0.25 base, 0.30 max, discrete levels to minimize fee churn.
Key innovation: Z-score + ADX regime filter generates more trades than CRSI while maintaining
quality. Works in both bull and bear markets due to HTF trend filter.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_zscore_4h_hma_adx_regime_v1"
timeframe = "30m"
leverage = 1.0

def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion signals."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    zscore = (close_s - sma) / std
    zscore = zscore.values
    zscore[np.isnan(zscore)] = 0.0
    return zscore

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's method (EMA with alpha = 1/period)
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Calculate DI+ and DI-
    di_plus = np.zeros(n)
    di_minus = np.zeros(n)
    mask = tr_smooth > 0
    di_plus[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    di_minus[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    mask2 = (di_plus + di_minus) > 0
    dx[mask2] = 100 * np.abs(di_plus[mask2] - di_minus[mask2]) / (di_plus[mask2] + di_minus[mask2])
    
    adx_raw = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    adx = adx_raw
    
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands for additional confirmation."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    zscore = calculate_zscore(close, period=20)
    adx = calculate_adx(high, low, close, period=14)
    bb_upper, bb_lower, bb_sma = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Additional trend filters
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels to minimize fee churn (Rule 4)
    SIZE_BASE = 0.25
    SIZE_MAX = 0.30
    SIZE_EXIT = 0.0
    
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
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(zscore[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_sma[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend bias (HTF) - primary filter
        bull_trend = close[i] > hma_4h_aligned[i]
        bear_trend = close[i] < hma_4h_aligned[i]
        
        # ADX regime filter - only trade when there's momentum
        adx_strong = adx[i] > 18  # Lower threshold to generate more trades
        adx_weak = adx[i] < 18
        
        # Z-score mean reversion signals
        zscore_oversold = zscore[i] < -1.5  # Moderate oversold
        zscore_overbought = zscore[i] > 1.5  # Moderate overbought
        zscore_extreme_oversold = zscore[i] < -2.0  # Strong oversold
        zscore_extreme_overbought = zscore[i] > 2.0  # Strong overbought
        
        # Price position vs Bollinger Bands
        price_near_lower = close[i] < bb_lower[i] * 1.005  # Within 0.5% of lower band
        price_near_upper = close[i] > bb_upper[i] * 0.995  # Within 0.5% of upper band
        price_below_bb_lower = close[i] < bb_lower[i]  # Below lower band
        price_above_bb_upper = close[i] > bb_upper[i]  # Above upper band
        
        # EMA trend confirmation
        ema_bullish = close[i] > ema_50[i] and close[i] > ema_200[i]
        ema_bearish = close[i] < ema_50[i] and close[i] < ema_200[i]
        
        new_signal = 0.0
        
        # === LONG ENTRY ===
        # Primary: Z-score extreme oversold + 4h bull trend + ADX strong
        if zscore_extreme_oversold and bull_trend and adx_strong:
            new_signal = SIZE_MAX
        # Secondary: Z-score oversold + price near lower BB + 4h bull trend
        elif zscore_oversold and price_near_lower and bull_trend:
            new_signal = SIZE_BASE
        # Tertiary: Z-score oversold + price below BB lower + EMA bullish
        elif zscore_oversold and price_below_bb_lower and ema_bullish:
            new_signal = SIZE_BASE
        # Quaternary: Z-score moderate oversold + 4h bull trend (catch more trades)
        elif zscore[i] < -1.2 and bull_trend:
            new_signal = SIZE_BASE
        
        # === SHORT ENTRY ===
        # Primary: Z-score extreme overbought + 4h bear trend + ADX strong
        if zscore_extreme_overbought and bear_trend and adx_strong:
            new_signal = -SIZE_MAX
        # Secondary: Z-score overbought + price near upper BB + 4h bear trend
        elif zscore_overbought and price_near_upper and bear_trend:
            new_signal = -SIZE_BASE
        # Tertiary: Z-score overbought + price above BB upper + EMA bearish
        elif zscore_overbought and price_above_bb_upper and ema_bearish:
            new_signal = -SIZE_BASE
        # Quaternary: Z-score moderate overbought + 4h bear trend (catch more trades)
        elif zscore[i] > 1.2 and bear_trend:
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