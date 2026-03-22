#!/usr/bin/env python3
"""
Experiment #190: 4h Volatility Spike Reversion + 1d HMA Trend + Fisher Transform

Hypothesis: After volatility spikes (panic/capitulation), prices tend to revert.
This strategy combines:
1. Volatility spike detection: ATR(7)/ATR(30) > 2.0 signals extreme vol
2. Mean reversion entry: Price at Bollinger Band extreme (2.5 std)
3. Fisher Transform confirmation: Catches reversal inflection points
4. 1d HMA trend filter: Only trade with higher-timeframe bias
5. ATR trailing stop: Protects against continued adverse moves

Why 4h might work for vol reversion:
- 4h captures panic moves without 15m/1h noise
- Vol spikes on 4h signal real capitulation, not just noise
- Fisher Transform excels at identifying reversal points in volatile markets
- 1d HMA prevents counter-trend trades in strong trends
- Conservative sizing (0.25) controls drawdown

Learning from failures:
- #183 (1h vol spike): Sharpe=-3.477 - 1h too noisy, 4h should be better
- #184 (4h KAMA chop): Sharpe=-2.908 - KAMA doesn't capture vol spikes well
- #189 (1h regime adaptive): Sharpe=-0.064 - too complex, simpler is better
- Pure trend following fails in 2022 crash and 2025 bear market
- Mean reversion WITH vol filter works better than pure mean reversion

Timeframe: 4h (REQUIRED for this experiment)
HTF: 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_vol_spike_fisher_1d_hma_bb_atr_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.5):
    """Calculate Bollinger Bands with configurable std multiplier."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    upper = sma + (std_mult * std)
    lower = sma - (std_mult * std)
    return upper.values, lower.values, sma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform for reversal detection.
    Transforms price into a Gaussian normal distribution for clearer signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = len(high)
    fisher = np.zeros(n)
    trigger = np.zeros(n)
    
    for i in range(period, n):
        # Calculate HL2 (typical price)
        hl2 = (high[i] + low[i]) / 2.0
        
        # Find highest high and lowest low over period
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        # Normalize price to 0-1 range
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        normalized = (hl2 - lowest) / range_val
        
        # Clamp to avoid log(0) or log(inf)
        normalized = np.clip(normalized, 0.001, 0.999)
        
        # Fisher transform formula
        fisher[i] = 0.5 * np.log((1 + normalized) / (1 - normalized))
        
        # Trigger line (1-period lag of fisher)
        if i > period:
            trigger[i] = fisher[i-1]
        else:
            trigger[i] = fisher[i]
    
    return fisher, trigger

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
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.5)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    rsi = calculate_rsi(close, 14)
    
    # Volatility ratio: ATR(7)/ATR(30)
    vol_ratio = np.divide(atr_7, atr_30, out=np.zeros_like(atr_7), where=atr_30!=0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_close = 0.0
    lowest_close = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(vol_ratio[i]) or np.isnan(bb_upper[i]) or np.isnan(fisher[i]):
            signals[i] = 0.0
            continue
        
        # === MULTI-TIMEFRAME TREND BIAS ===
        # 1d HMA = higher timeframe trend bias
        bull_trend_1d = close[i] > hma_1d_aligned[i]
        bear_trend_1d = close[i] < hma_1d_aligned[i]
        
        # === VOLATILITY SPIKE DETECTION ===
        # ATR(7)/ATR(30) > 2.0 = extreme volatility (panic/capitulation)
        vol_spike = vol_ratio[i] > 2.0
        
        # === BOLLINGER BAND POSITION ===
        # Price at lower band = oversold (long candidate)
        # Price at upper band = overbought (short candidate)
        at_lower_bb = close[i] <= bb_lower[i]
        at_upper_bb = close[i] >= bb_upper[i]
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        # Fisher crosses above -1.5 = bullish reversal
        # Fisher crosses below +1.5 = bearish reversal
        fisher_bullish = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        fisher_bearish = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        
        # Also check Fisher extreme levels for mean reversion
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi[i] < 35
        rsi_overbought = rsi[i] > 65
        
        new_signal = 0.0
        
        # === ENTRY CONDITIONS ===
        # Long: 1d bullish OR neutral + vol spike + at lower BB + (Fisher bullish OR extreme low)
        # More flexible: allow longs in bear trend if vol spike is extreme
        if vol_spike and at_lower_bb:
            if fisher_bullish or fisher_extreme_low:
                # Prefer longs in bull trend, but allow in bear if RSI confirms
                if bull_trend_1d or (bear_trend_1d and rsi_oversold):
                    new_signal = SIZE_BASE
        
        # Short: 1d bearish OR neutral + vol spike + at upper BB + (Fisher bearish OR extreme high)
        if vol_spike and at_upper_bb:
            if fisher_bearish or fisher_extreme_high:
                # Prefer shorts in bear trend, but allow in bull if RSI confirms
                if bear_trend_1d or (bull_trend_1d and rsi_overbought):
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and new_signal != 0.0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.5 * ATR below highest close
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.5 * ATR above lowest close
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # Also check stoploss if no new signal (exit position)
        if in_position and new_signal == 0.0:
            if position_side > 0:
                if close[i] > highest_close:
                    highest_close = close[i]
                stoploss_price = highest_close - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0
            
            if position_side < 0:
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                stoploss_price = lowest_close + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0
        
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
            # else: maintaining same position direction, update extremes
            elif position_side > 0 and close[i] > highest_close:
                highest_close = close[i]
            elif position_side < 0 and (lowest_close == 0.0 or close[i] < lowest_close):
                lowest_close = close[i]
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