#!/usr/bin/env python3
"""
Experiment #189: 1h Regime-Adaptive Strategy with 4h HMA Trend Filter

Hypothesis: 1h timeframe needs regime-adaptive logic because crypto alternates
between trending and ranging markets. Use ADX to detect regime:
- ADX > 25: Trend-following mode (HMA/EMA crossover with 4h bias)
- ADX < 20: Mean-reversion mode (RSI extremes + Bollinger Bands)
- ADX 20-25: No-trade zone (avoid whipsaw during regime transitions)

4h HMA provides higher-timeframe trend bias to avoid counter-trend trades.
This should work in both 2021-2024 (trending) and 2025 (bear/range) periods.

Why 1h with regime-adaptive might work:
- 1h captures intraday moves but needs filters to avoid noise
- ADX regime detection switches logic based on market state
- 4h HMA filter prevents fighting the higher-timeframe trend
- Mean-reversion mode catches reversals in 2025 bear/range market
- Trend mode captures momentum in 2021 bull and 2023 recovery

Learning from failures:
- #177 (1h KAMA): Sharpe=-0.061 - pure trend failed in ranges
- #183 (1h vol spike): Sharpe=-3.477 - mean-reversion alone failed
- #187 (15m Supertrend): Sharpe=-1.239 - too noisy, wrong regime
- Key insight: Need BOTH trend and mean-reversion, switched by ADX

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_adaptive_4h_hma_adx_bb_rsi_atr_v1"
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

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def calculate_fisher_transform(high, low, period=9):
    """Calculate Ehlers Fisher Transform for reversal detection."""
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val == 0:
            range_val = 1e-10
        
        # Normalize price to -1 to +1 range
        normalized = 0.6667 * ((hl2 - lowest) / range_val - 0.5) + 0.67 * np.clip(fisher[i-1], -0.99, 0.99)
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        if i > 0:
            trigger[i] = fisher[i-1]
    
    return fisher, trigger

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
    rsi = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    ema_21 = calculate_ema(close, 21)
    ema_50 = calculate_ema(close, 50)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.125
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx[i]) or np.isnan(rsi[i]) or np.isnan(bb_upper[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 4h HMA = higher timeframe trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION ===
        # ADX > 25 = trending market (use trend-following logic)
        # ADX < 20 = ranging market (use mean-reversion logic)
        # ADX 20-25 = transition zone (no new entries)
        trend_regime = adx[i] > 25
        range_regime = adx[i] < 20
        
        # === TREND-FOLLOWING SIGNALS (ADX > 25) ===
        trend_long = False
        trend_short = False
        
        if trend_regime:
            # Long: 4h bullish + EMA21 > EMA50 + price > EMA21
            if bull_trend_4h and ema_21[i] > ema_50[i] and close[i] > ema_21[i]:
                trend_long = True
            
            # Short: 4h bearish + EMA21 < EMA50 + price < EMA21
            if bear_trend_4h and ema_21[i] < ema_50[i] and close[i] < ema_21[i]:
                trend_short = True
        
        # === MEAN-REVERSION SIGNALS (ADX < 20) ===
        mr_long = False
        mr_short = False
        
        if range_regime:
            # Long: RSI < 30 + price < BB lower + 4h bullish bias preferred
            if rsi[i] < 30 and close[i] < bb_lower[i]:
                mr_long = True
            
            # Short: RSI > 70 + price > BB upper + 4h bearish bias preferred
            if rsi[i] > 70 and close[i] > bb_upper[i]:
                mr_short = True
            
            # Fisher Transform reversals (works well in ranges)
            # Long: Fisher crosses above -1.5 from below
            if fisher[i] > -1.5 and fisher_trigger[i] <= -1.5:
                mr_long = True
            
            # Short: Fisher crosses below +1.5 from above
            if fisher[i] < 1.5 and fisher_trigger[i] >= 1.5:
                mr_short = True
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Trend regime: use trend-following signals
        if trend_regime:
            if trend_long:
                new_signal = SIZE_BASE
            elif trend_short:
                new_signal = -SIZE_BASE
        
        # Range regime: use mean-reversion signals
        elif range_regime:
            if mr_long:
                new_signal = SIZE_BASE
            elif mr_short:
                new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and new_signal != 0.0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            elif position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === STOPLOSS WHEN FLATTING ===
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Confirm flat on stoploss
            
            elif position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Confirm flat on stoploss
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals