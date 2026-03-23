#!/usr/bin/env python3
"""
Experiment #001: 4h Primary + 1d/1w HTF — Fisher Transform + Donchian Breakout + Regime Adaptive

Hypothesis: Based on research showing Ehlers Fisher Transform excels at catching reversals in 
bear/range markets (better than RSI), combined with Donchian breakouts for trend confirmation.
Adding 1w HMA for major trend bias improves win rate on BTC/ETH which fail simple trend strategies.

Key innovations:
1. EHLERS FISHER TRANSFORM: period=9, catches reversals at extremes (-1.5/+1.5 thresholds)
   Superior to RSI in bear markets per quantitative literature
2. DONCHIAN CHANNEL: 20-period breakout for trend confirmation, reduces false signals
3. 1W HMA: Major trend bias — only trade with weekly trend for higher win rate
4. 1D HMA: Intermediate trend filter
5. REGIME ADAPTIVE: Different entry thresholds for bull/bear/range based on 1w HMA slope
6. ASYMMETRIC SIZING: 0.30 with trend, 0.20 against (risk management)

Why 4h works:
- Targets 20-50 trades/year (fee-efficient per Rule 10)
- Less noise than 1h, more signals than 12h
- Proven in baseline strategies

Entry conditions (LOOSE enough to generate 30+ trades):
- Long: Fisher < -1.2 + price > Donchian_mid + 1w HMA bullish OR Fisher < -1.5 (strong reversal)
- Short: Fisher > +1.2 + price < Donchian_mid + 1w HMA bearish OR Fisher > +1.5 (strong reversal)
- Trend confirmation: 1d HMA slope agrees with 1w HMA

Position size: 0.25-0.30 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_donchian_regime_1d1w_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 0.67 * (price - lowest) / (highest - lowest) - 0.33
    Catches reversals at extremes: < -1.5 = oversold, > +1.5 = overbought
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_prev = np.zeros(n)
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        price_range = highest - lowest
        if price_range < 1e-10:
            price_range = 1e-10
        
        # Calculate X value
        x = 0.67 * (close[i] - lowest) / price_range - 0.33
        
        # Clamp X to avoid division by zero in log
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
        
        # Smooth with previous value (Ehlers method)
        if i > period:
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher_prev[i-1]
        
        fisher_prev[i] = fisher[i]
    
    return fisher

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low, mid)."""
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (highest + lowest) / 2.0
    return highest, lowest, mid

def calculate_rsi(close, period=14):
    """Calculate RSI for additional filter."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for intermediate trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 1w HMA for major trend bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    
    fisher = calculate_fisher_transform(high, low, close, period=9)
    
    donchian_high, donchian_low, donchian_mid = calculate_donchian(high, low, period=20)
    
    rsi_14 = calculate_rsi(close, period=14)
    
    # 4h HMA for local trend
    hma_4h = calculate_hma(close, period=21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE_WITH_TREND = 0.30  # Full size when aligned with 1w trend
    POSITION_SIZE_COUNTER = 0.20     # Reduced size against 1w trend
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(atr_14[i]) or np.isnan(fisher[i]) or np.isnan(donchian_mid[i]):
            continue
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h[i]):
            continue
        if atr_14[i] == 0:
            continue
        
        # === 1W MACRO TREND BIAS (most important) ===
        hma_1w_slope_bull = hma_1w_aligned[i] > hma_1w_aligned[i-5] if i >= 5 else False
        hma_1w_slope_bear = hma_1w_aligned[i] < hma_1w_aligned[i-5] if i >= 5 else False
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        hma_1d_slope_bull = hma_1d_aligned[i] > hma_1d_aligned[i-3] if i >= 3 else False
        hma_1d_slope_bear = hma_1d_aligned[i] < hma_1d_aligned[i-3] if i >= 3 else False
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === 4H LOCAL TREND ===
        price_above_hma_4h = close[i] > hma_4h[i]
        price_below_hma_4h = close[i] < hma_4h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold_strong = fisher[i] < -1.5  # Strong reversal signal
        fisher_oversold_moderate = fisher[i] < -1.2 and fisher[i] >= -1.5
        fisher_overbought_strong = fisher[i] > 1.5  # Strong reversal signal
        fisher_overbought_moderate = fisher[i] > 1.2 and fisher[i] <= 1.5
        
        # Fisher cross signals (momentum)
        fisher_cross_up = fisher[i] > -1.5 and fisher[i-1] <= -1.5 if i > 0 else False
        fisher_cross_down = fisher[i] < 1.5 and fisher[i-1] >= 1.5 if i > 0 else False
        
        # === DONCHIAN POSITION ===
        price_above_donchian_mid = close[i] > donchian_mid[i]
        price_below_donchian_mid = close[i] < donchian_mid[i]
        
        # Near Donchian breakout (within 1%)
        price_near_donchian_high = close[i] > donchian_high[i] * 0.99
        price_near_donchian_low = close[i] < donchian_low[i] * 1.01
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === REGIME DETECTION ===
        # Bull regime: 1w HMA bullish + price above 1w HMA
        is_bull_regime = hma_1w_slope_bull and price_above_hma_1w
        
        # Bear regime: 1w HMA bearish + price below 1w HMA
        is_bear_regime = hma_1w_slope_bear and price_below_hma_1w
        
        # Range regime: mixed signals
        is_range_regime = not is_bull_regime and not is_bear_regime
        
        # === ADAPTIVE ENTRY LOGIC ===
        new_signal = 0.0
        position_size = POSITION_SIZE_WITH_TREND
        
        # --- BULL REGIME: Prefer longs, shorts only on strong reversal ---
        if is_bull_regime:
            # Long: Fisher oversold + price above Donchian mid + 1d confirmation
            if (fisher_oversold_strong or (fisher_oversold_moderate and rsi_oversold)):
                if price_above_donchian_mid and (price_above_hma_1d or fisher_cross_up):
                    new_signal = POSITION_SIZE_WITH_TREND
                    position_size = POSITION_SIZE_WITH_TREND
            
            # Short only on VERY strong reversal (counter-trend, reduced size)
            elif fisher_overbought_strong and rsi_overbought:
                if price_near_donchian_high:
                    new_signal = -POSITION_SIZE_COUNTER
                    position_size = POSITION_SIZE_COUNTER
        
        # --- BEAR REGIME: Prefer shorts, longs only on strong reversal ---
        elif is_bear_regime:
            # Short: Fisher overbought + price below Donchian mid + 1d confirmation
            if (fisher_overbought_strong or (fisher_overbought_moderate and rsi_overbought)):
                if price_below_donchian_mid and (price_below_hma_1d or fisher_cross_down):
                    new_signal = -POSITION_SIZE_WITH_TREND
                    position_size = POSITION_SIZE_WITH_TREND
            
            # Long only on VERY strong reversal (counter-trend, reduced size)
            elif fisher_oversold_strong and rsi_oversold:
                if price_near_donchian_low:
                    new_signal = POSITION_SIZE_COUNTER
                    position_size = POSITION_SIZE_COUNTER
        
        # --- RANGE REGIME: Mean reversion both directions ---
        else:
            # Long: Fisher oversold + near Donchian low
            if fisher_oversold_moderate or fisher_oversold_strong:
                if price_near_donchian_low or (price_below_donchian_mid and rsi_oversold):
                    new_signal = POSITION_SIZE_COUNTER  # Reduced size in range
                    position_size = POSITION_SIZE_COUNTER
            
            # Short: Fisher overbought + near Donchian high
            elif fisher_overbought_moderate or fisher_overbought_strong:
                if price_near_donchian_high or (price_above_donchian_mid and rsi_overbought):
                    new_signal = -POSITION_SIZE_COUNTER  # Reduced size in range
                    position_size = POSITION_SIZE_COUNTER
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if regime changes to strong bear
        if in_position and position_side > 0:
            if is_bear_regime and hma_1d_slope_bear:
                new_signal = 0.0
        
        # Exit short if regime changes to strong bull
        if in_position and position_side < 0:
            if is_bull_regime and hma_1d_slope_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals