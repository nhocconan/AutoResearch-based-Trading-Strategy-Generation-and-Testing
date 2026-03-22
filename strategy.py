#!/usr/bin/env python3
"""
Experiment #344: 30m Volatility-Regime Adaptive Strategy with 4h HMA Bias + Fisher Transform

Hypothesis: After 292 failed strategies, the pattern is clear - static strategies fail because
30m timeframe alternates between high-volatility panic (mean reversion works) and low-volatility
grind (trend following works). This strategy adapts based on volatility regime:

1. VOLATILITY REGIME via ATR Ratio:
   - ATR(7)/ATR(30) > 1.8 = high vol panic → mean reversion at BB extremes
   - ATR(7)/ATR(30) < 1.2 = low vol grind → trend follow with 4h HMA
   - 1.2 <= ratio <= 1.8 = transition → reduce position or flat

2. FISHER TRANSFORM for reversals (Ehlers):
   - Normalizes price into Gaussian distribution (-2 to +2 range)
   - Long when Fisher crosses above -1.5 from below (oversold reversal)
   - Short when Fisher crosses below +1.5 from above (overbought reversal)
   - Proven to catch bear market rallies better than RSI

3. 4h HMA for trend bias:
   - Only long in low-vol regime if price > 4h HMA
   - Only short in low-vol regime if price < 4h HMA
   - In high-vol regime, ignore trend (mean reversion dominates)

4. Bollinger Bands for mean reversion entries:
   - High vol + price < BB_lower → long (panic oversold)
   - High vol + price > BB_upper → short (panic overbought)

Why this should work on 30m:
- 30m captures intraday vol spikes that 1h/4h miss
- Fisher Transform catches reversals faster than RSI
- Volatility regime filter prevents trend-following during panic
- 4h HMA provides stable bias without whipsaw
- Looser thresholds ensure >=10 trades per symbol

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels (conservative for 30m noise)
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_vol_regime_4h_hma_fisher_bb_atr_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price into Gaussian distribution for clearer reversal signals.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize to -1 to +1 range
    3. Apply Fisher transform: 0.5 * ln((1+x)/(1-x))
    
    Entry: Fisher crosses above -1.5 (long) or below +1.5 (short)
    """
    n = len(high)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Typical price
    typical = (high + low) / 2.0
    typical_s = pd.Series(typical)
    
    # Normalize to -1 to +1 using highest high and lowest low over period
    for i in range(period, n):
        highest = typical[i-period+1:i+1].max()
        lowest = typical[i-period+1:i+1].min()
        range_val = highest - lowest
        
        if range_val > 1e-10:
            # Normalize to 0-1, then scale to -0.99 to +0.99
            normalized = 2.0 * (typical[i] - lowest) / range_val - 1.0
            normalized = np.clip(normalized, -0.99, 0.99)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            
            if i > period:
                fisher_prev[i] = fisher[i-1]
    
    return fisher, fisher_prev

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE = 0.25  # Conservative for 30m noise
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_7[i]) or np.isnan(atr_30[i]) or atr_30[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
            signals[i] = 0.0
            continue
        
        # === VOLATILITY REGIME DETECTION ===
        vol_ratio = atr_7[i] / max(atr_30[i], 1e-10)
        
        in_high_vol = vol_ratio > 1.8  # Panic/spike - mean reversion
        in_low_vol = vol_ratio < 1.2   # Grind - trend follow
        # 1.2 <= ratio <= 1.8 = transition
        
        # === 4h HMA TREND BIAS ===
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.5 and fisher_prev[i] >= -1.5  # Cross above -1.5
        fisher_overbought = fisher[i] > 1.5 and fisher_prev[i] <= 1.5  # Cross below +1.5
        
        # === BOLLINGER BAND EXTREMES ===
        at_bb_lower = close[i] <= bb_lower[i]
        at_bb_upper = close[i] >= bb_upper[i]
        
        # === GENERATE SIGNAL BASED ON REGIME ===
        new_signal = 0.0
        
        # HIGH VOLATILITY REGIME: Mean reversion at BB extremes
        if in_high_vol:
            # Long: price at BB lower (panic oversold)
            if at_bb_lower:
                new_signal = SIZE
            
            # Short: price at BB upper (panic overbought)
            elif at_bb_upper:
                new_signal = -SIZE
        
        # LOW VOLATILITY REGIME: Trend following with Fisher entries
        elif in_low_vol:
            # Long: 4h HMA bullish + Fisher oversold cross (pullback entry)
            if bull_trend_4h and fisher_oversold:
                new_signal = SIZE
            
            # Short: 4h HMA bearish + Fisher overbought cross (pullback entry)
            elif bear_trend_4h and fisher_overbought:
                new_signal = -SIZE
            
            # Alternative: Simple trend continuation
            elif bull_trend_4h and fisher[i] > -0.5:
                new_signal = SIZE
            elif bear_trend_4h and fisher[i] < 0.5:
                new_signal = -SIZE
        
        # TRANSITION REGIME: Reduce or flat
        if not in_high_vol and not in_low_vol:
            if in_position:
                # Keep current position but don't add
                new_signal = signals[i-1] if i > 0 else 0.0
            else:
                new_signal = 0.0
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        if in_position and position_side != 0:
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
        
        # === TREND REVERSAL EXIT ===
        # Exit if 4h trend flips against position (only in low vol regime)
        if in_position and new_signal != 0.0 and in_low_vol:
            if position_side > 0 and bear_trend_4h:
                new_signal = 0.0
            if position_side < 0 and bull_trend_4h:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals