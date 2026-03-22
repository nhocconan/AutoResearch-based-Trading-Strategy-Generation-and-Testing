#!/usr/bin/env python3
"""
Experiment #202: 4h Asymmetric Regime + 1d HMA Trend + ADX Filter + ATR Stop

Hypothesis: Different market regimes require different entry logic.
- Bear regime (ADX>25 + price<SMA50): Only short rallies to EMA21, avoid longs
- Bull regime (ADX>25 + price>SMA50): Only long pullbacks to EMA21, avoid shorts  
- Range regime (ADX<20): Mean revert at Bollinger Band extremes

This asymmetric approach should reduce whipsaw losses in 2022-style crashes
while capturing trends in bull markets. Combined with 1d HMA for higher TF bias
and ATR trailing stops for risk management.

Why this might work:
- Asymmetric regime logic mentioned in research as promising for BTC/ETH
- Avoids counter-trend trades in strong bear markets (major failure mode)
- 4h timeframe balances signal frequency with noise reduction
- 1d HMA provides stable higher-timeframe trend bias
- ADX hysteresis (enter 25, exit 18) prevents regime flip-flopping
- Conservative sizing (0.25) controls drawdown

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_asymmetric_regime_1d_hma_adx_atr_v1"
timeframe = "4h"
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength."""
    n = len(close)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    plus_dm_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    tr_s = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    tr_s = np.where(tr_s == 0, 1e-10, tr_s)
    
    plus_di = 100 * plus_dm_s / tr_s
    minus_di = 100 * minus_dm_s / tr_s
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
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

def calculate_ema(close, period=21):
    """Calculate Exponential Moving Average."""
    close_s = pd.Series(close)
    ema = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    return ema.values

def calculate_sma(close, period=50):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def calculate_rsi(close, period=14):
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HTF indicators
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 4h indicators
    atr = calculate_atr(high, low, close, 14)
    adx = calculate_adx(high, low, close, 14)
    ema_21 = calculate_ema(close, 21)
    sma_50 = calculate_sma(close, 50)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    
    # Track regime state with hysteresis
    prev_regime = 0  # 0=unknown, 1=bull, -1=bear, 2=range
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(ema_21[i]) or np.isnan(sma_50[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME TREND BIAS ===
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION WITH HYSTERESIS ===
        # Enter trending regime at ADX > 25, exit at ADX < 18
        # This prevents rapid regime flip-flopping
        
        current_adx = adx[i]
        
        if prev_regime == 0:
            # Initial regime detection
            if current_adx > 25:
                if close[i] > sma_50[i]:
                    current_regime = 1  # Bull trend
                else:
                    current_regime = -1  # Bear trend
            elif current_adx < 20:
                current_regime = 2  # Range
            else:
                current_regime = prev_regime
        elif prev_regime == 1 or prev_regime == -1:
            # In trending regime - exit only when ADX drops below 18
            if current_adx < 18:
                current_regime = 2  # Switch to range
            else:
                # Stay in trend, but check if bull/bear flipped
                if close[i] > sma_50[i]:
                    current_regime = 1
                else:
                    current_regime = -1
        else:  # prev_regime == 2 (range)
            # In range regime - enter trend only when ADX > 25
            if current_adx > 25:
                if close[i] > sma_50[i]:
                    current_regime = 1
                else:
                    current_regime = -1
            else:
                current_regime = 2
        
        prev_regime = current_regime
        
        # === ASYMMETRIC ENTRY LOGIC BY REGIME ===
        new_signal = 0.0
        
        if current_regime == 1:  # BULL TREND
            # Only long pullbacks, avoid shorts
            # Entry: price pulls back to EMA21 + RSI < 50 + 1d bullish
            pullback_long = (close[i] <= ema_21[i] * 1.005) and (close[i] >= ema_21[i] * 0.995)
            rsi_oversold = rsi[i] < 50
            
            if pullback_long and rsi_oversold and bull_trend_1d:
                new_signal = SIZE_BASE
        
        elif current_regime == -1:  # BEAR TREND
            # Only short rallies, avoid longs
            # Entry: price rallies to EMA21 + RSI > 50 + 1d bearish
            rally_short = (close[i] >= ema_21[i] * 0.995) and (close[i] <= ema_21[i] * 1.005)
            rsi_overbought = rsi[i] > 50
            
            if rally_short and rsi_overbought and bear_trend_1d:
                new_signal = -SIZE_BASE
        
        elif current_regime == 2:  # RANGE
            # Mean revert at Bollinger Band extremes
            # Long at lower band + RSI < 35
            # Short at upper band + RSI > 65
            at_lower_band = close[i] <= bb_lower[i] * 1.005
            at_upper_band = close[i] >= bb_upper[i] * 0.995
            rsi_extreme_low = rsi[i] < 35
            rsi_extreme_high = rsi[i] > 65
            
            if at_lower_band and rsi_extreme_low:
                new_signal = SIZE_BASE
            elif at_upper_band and rsi_extreme_high:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals