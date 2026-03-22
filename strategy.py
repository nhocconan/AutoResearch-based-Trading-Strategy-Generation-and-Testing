#!/usr/bin/env python3
"""
Experiment #141: 1h Asymmetric Regime + 4h HMA Trend + ADX Hysteresis + ATR Stop

Hypothesis: Based on research showing bear market strategies need regime-adaptive logic.
Key insight from 200+ failed experiments: simple trend following gets whipsawed in 2022,
pure mean reversion fails in strong trends. This strategy uses:

1. ASYMMETRIC REGIME (from research): 
   - ADX>25 + price<SMA50 = bear regime (only short retraces to EMA21)
   - ADX<20 = range regime (mean revert at BB bounds)
   - Hysteresis: enter trend mode at 25, exit at 18 (prevents chop)

2. MULTI-TIMEFRAME: 4h HMA(21) for trend bias (proven to 2x Sharpe in winning strategies)

3. DUAL ENTRY LOGIC:
   - Trend mode: pullback to EMA21 in direction of 4h HMA
   - Range mode: fade BB(20,2.0) extremes with RSI(14) confirmation

4. RISK: ATR(14) trailing stop at 2.5*ATR, discrete position sizing 0.20-0.35

Why this might beat baseline (Sharpe=0.478):
- Regime adaptation prevents trend-following losses in 2022 crash
- 4h HMA filter avoids counter-trend trades (major source of losses)
- ADX hysteresis reduces whipsaw between regimes
- 1h timeframe balances signal frequency vs noise

Timeframe: 1h (REQUIRED for experiment #141)
HTF: 4h via mtf_data helper (call ONCE before loop)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_asymmetric_regime_4h_hma_adx_hyst_atr_v1"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    n = len(close)
    adx = np.zeros(n)
    adx[:] = np.nan
    
    if n < period * 2:
        return adx
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        plus_dm[i] = max(0, high[i] - high[i-1]) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(0, low[i-1] - low[i]) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = tr_s > 0
    plus_di[mask] = 100 * plus_dm_s[mask] / tr_s[mask]
    minus_di[mask] = 100 * minus_dm_s[mask] / tr_s[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    adx_series = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx_series.values
    
    return adx

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    pct_b = (close - lower) / (upper - lower)
    return upper, lower, pct_b

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

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
    
    # Calculate 1h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    ema21 = calculate_ema(close, 21)
    sma50 = calculate_sma(close, 50)
    bb_upper, bb_lower, pct_b = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.35
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    # ADX hysteresis state
    in_trend_mode = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(ema21[i]) or np.isnan(sma50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi[i]) or np.isnan(pct_b[i]):
            signals[i] = 0.0
            continue
        
        # === ADX REGIME WITH HYSTERESIS ===
        # Enter trend mode when ADX > 25, exit when ADX < 18
        if adx[i] > 25:
            in_trend_mode = True
        elif adx[i] < 18:
            in_trend_mode = False
        
        # === 4H TREND BIAS ===
        bull_4h = close[i] > hma_4h_aligned[i]
        bear_4h = close[i] < hma_4h_aligned[i]
        
        # === BEAR REGIME DETECTION ===
        # ADX > 20 + price < SMA50 = bear market (only short retraces)
        bear_regime = adx[i] > 20 and close[i] < sma50[i]
        
        new_signal = 0.0
        
        # === TREND MODE ENTRIES (ADX hysteresis > 18) ===
        if in_trend_mode:
            # Long: 4h bullish + pullback to EMA21 + RSI not overbought
            if bull_4h and close[i] <= ema21[i] * 1.002 and rsi[i] < 70:
                new_signal = SIZE_BASE
                # Strong signal if also above SMA50
                if close[i] > sma50[i]:
                    new_signal = SIZE_STRONG
            
            # Short: 4h bearish + rally to EMA21 + RSI not oversold
            if bear_4h and close[i] >= ema21[i] * 0.998 and rsi[i] > 30:
                new_signal = -SIZE_BASE
                # Strong signal if also below SMA50
                if close[i] < sma50[i]:
                    new_signal = -SIZE_STRONG
        
        # === RANGE MODE ENTRIES (ADX < 18) ===
        else:
            # Long: price at BB lower + RSI oversold + 4h not strongly bearish
            if pct_b[i] < 0.1 and rsi[i] < 35:
                if not bear_4h or rsi[i] < 25:  # Allow long even in bear if RSI very low
                    new_signal = SIZE_BASE
            
            # Short: price at BB upper + RSI overbought + 4h not strongly bullish
            if pct_b[i] > 0.9 and rsi[i] > 65:
                if not bull_4h or rsi[i] > 75:  # Allow short even in bull if RSI very high
                    new_signal = -SIZE_BASE
        
        # === BEAR REGIME SPECIAL LOGIC ===
        # In bear regime, prioritize shorts on retraces, limit longs
        if bear_regime:
            # Only short on retraces to EMA in bear regime
            if bear_4h and close[i] >= ema21[i] * 0.995 and rsi[i] > 35:
                new_signal = -SIZE_STRONG
            # Limit longs to extreme oversold only
            if new_signal > 0 and rsi[i] > 40:
                new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side > 0:
            if close[i] > highest_close:
                highest_close = close[i]
            stoploss_price = highest_close - 2.5 * atr[i]
            if close[i] < stoploss_price:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if lowest_close == 0.0 or close[i] < lowest_close:
                lowest_close = close[i]
            stoploss_price = lowest_close + 2.5 * atr[i]
            if close[i] > stoploss_price:
                new_signal = 0.0
        
        # Update position tracking
        if new_signal != 0.0 and not in_position:
            in_position = True
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        elif new_signal != 0.0 and in_position and np.sign(new_signal) != position_side:
            position_side = np.sign(new_signal)
            entry_price = close[i]
            highest_close = close[i] if position_side > 0 else 0.0
            lowest_close = close[i] if position_side < 0 else 0.0
        elif new_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
            entry_price = 0.0
            highest_close = 0.0
            lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals