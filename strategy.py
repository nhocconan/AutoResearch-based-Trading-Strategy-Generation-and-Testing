#!/usr/bin/env python3
"""
Experiment #067: 1d Primary + 1w HTF — Fisher Transform + Vol Spike Reversion

Hypothesis: Previous 1d strategies failed because Connors RSI + Choppiness is overused
and doesn't work well on daily timeframes. This strategy uses DIFFERENT edges:

1. EHLERS FISHER TRANSFORM (period=9): Proven reversal indicator for bear markets
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
   - Works better than RSI in trending/bear markets

2. VOL SPIKE DETECTION (ATR(7)/ATR(30) > 2.0): Captures panic extremes
   - High vol ratio = panic selling/buying = mean reversion opportunity
   - Exit when vol ratio < 1.3 (vol crush)

3. 1W HMA(21) SLOPE: Major trend bias from weekly timeframe
   - Slope > 0 = prefer longs (but still allow shorts on vol spikes)
   - Slope < 0 = prefer shorts (bear market bias for 2025)

4. CHOPPINESS INDEX REGIME SWITCH:
   - CHOP > 61.8 = range (use Fisher mean reversion)
   - CHOP < 38.2 = trend (use Fisher with trend bias only)

5. ATR(14) trailing stoploss at 2.5x (wider for 1d timeframe)

Why this should work:
- Fisher Transform catches reversals better than RSI in bear markets
- Vol spike detection captures panic extremes (2022 crash, 2025 decline)
- 1w HTF provides major trend context without over-filtering
- 1d timeframe naturally limits trades to 20-40/year
- Simpler entry logic = more trades = better statistics

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.30 discrete (conservative for daily swings)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 25-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_volspike_chop_1w_v1"
timeframe = "1d"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals better than RSI in trending markets.
    """
    hl2 = (high + low) / 2
    
    # Normalize price to range -1 to +1
    lowest_low = pd.Series(hl2).rolling(window=period, min_periods=period).min().values
    highest_high = pd.Series(hl2).rolling(window=period, min_periods=period).max().values
    
    # Avoid division by zero
    range_val = highest_high - lowest_low
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    normalized = 2 * (hl2 - lowest_low) / range_val - 1
    
    # Clamp to avoid extreme values
    normalized = np.clip(normalized, -0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag of fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Choppiness Index (CHOP) - measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range-bound
    CHOP < 38.2 = trending
    """
    atr = calculate_atr(high, low, close, period)
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Avoid division by zero
    range_val = highest_high - lowest_low
    range_val = np.where(range_val < 1e-10, 1e-10, range_val)
    
    # Sum of ATR over period
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    # CHOP formula
    chop = 100 * np.log10(atr_sum / range_val) / np.log10(period)
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope over lookback period (percentage change)."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_vol_spike_ratio(atr_short=7, atr_long=30, high=None, low=None, close=None):
    """Calculate ATR ratio for vol spike detection."""
    atr_s = calculate_atr(high, low, close, atr_short)
    atr_l = calculate_atr(high, low, close, atr_long)
    
    # Avoid division by zero
    ratio = np.where(atr_l < 1e-10, 1.0, atr_s / atr_l)
    return ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    vol_ratio = calculate_vol_spike_ratio(7, 30, high, low, close)
    
    # 1d HMA for trend filter
    hma_1d_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(vol_ratio[i]):
            continue
        
        if np.isnan(hma_1d_21[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # HMA slope > 0 = bullish bias (prefer longs)
        # HMA slope < 0 = bearish bias (prefer shorts)
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = range (mean reversion mode)
        # CHOP < 38.2 = trend (trend following mode)
        # 38.2 < CHOP < 61.8 = neutral (allow both)
        regime_range = chop_14[i] > 61.8
        regime_trend = chop_14[i] < 38.2
        
        # === VOL SPIKE DETECTION ===
        # vol_ratio > 2.0 = panic/extreme vol (mean reversion opportunity)
        # vol_ratio < 1.3 = vol crush (exit signal)
        vol_spike = vol_ratio[i] > 2.0
        vol_crush = vol_ratio[i] < 1.3
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above -1.5 = oversold reversal (long)
        # Fisher crosses below +1.5 = overbought reversal (short)
        fisher_cross_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_cross_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Fisher extreme levels
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in weak signals
        if not vol_spike and not fisher_oversold and not fisher_overbought:
            current_size = REDUCED_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        # Mode 1: Vol spike + Fisher oversold (panic reversal)
        if vol_spike and fisher_oversold:
            new_signal = current_size
        
        # Mode 2: Fisher cross long + range regime (mean reversion)
        elif fisher_cross_long and regime_range:
            new_signal = current_size
        
        # Mode 3: Fisher cross long + bullish 1w trend (trend following)
        elif fisher_cross_long and trend_1w_bullish and not regime_trend:
            new_signal = current_size
        
        # Mode 4: Price above 1d HMA + Fisher neutral-bullish (trend continuation)
        elif close[i] > hma_1d_21[i] and fisher[i] > -0.5 and fisher[i] < 1.0:
            if trend_1w_bullish or regime_range:
                new_signal = REDUCED_SIZE
        
        # SHORT ENTRIES
        # Mode 1: Vol spike + Fisher overbought (panic reversal)
        if vol_spike and fisher_overbought:
            new_signal = -current_size
        
        # Mode 2: Fisher cross short + range regime (mean reversion)
        elif fisher_cross_short and regime_range:
            new_signal = -current_size
        
        # Mode 3: Fisher cross short + bearish 1w trend (trend following)
        elif fisher_cross_short and trend_1w_bearish and not regime_trend:
            new_signal = -current_size
        
        # Mode 4: Price below 1d HMA + Fisher neutral-bearish (trend continuation)
        elif close[i] < hma_1d_21[i] and fisher[i] > -1.0 and fisher[i] < 0.5:
            if trend_1w_bearish or regime_range:
                new_signal = -REDUCED_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 90 bars (~90 days on 1d), allow weaker entry
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if fisher_oversold and close[i] > hma_1d_21[i]:
                new_signal = REDUCED_SIZE
            elif fisher_overbought and close[i] < hma_1d_21[i]:
                new_signal = -REDUCED_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === VOL CRUSH EXIT ===
        # Exit when vol spike reverses (vol crush)
        vol_exit = False
        if in_position and vol_crush:
            vol_exit = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1w trend reverses strongly bearish
            if position_side > 0 and trend_1w_bearish and chop_14[i] < 38.2:
                trend_reversal = True
            # Exit short if 1w trend reverses strongly bullish
            if position_side < 0 and trend_1w_bullish and chop_14[i] < 38.2:
                trend_reversal = True
        
        # Apply stoploss or exits
        if stoploss_triggered or vol_exit or trend_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals