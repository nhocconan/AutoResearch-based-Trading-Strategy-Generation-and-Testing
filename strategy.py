#!/usr/bin/env python3
"""
Experiment #126: 12h Primary + 1d HTF — Choppiness Index Regime + Fisher Transform

Hypothesis: Previous 12h strategies failed because they used单一 approach (either pure trend
or pure mean-reversion). Market regime changes between trending and ranging, requiring
adaptive logic. This strategy uses:

1) Choppiness Index (CHOP) to detect regime:
   - CHOP > 61.8 = ranging market → use Fisher Transform mean reversion
   - CHOP < 38.2 = trending market → use Donchian breakout
   - 38.2 <= CHOP <= 61.8 = neutral → stay flat (avoid whipsaws)

2) Fisher Transform (Ehlers) for mean-reversion entries in choppy markets:
   - Fisher crosses above -1.5 from below → long (oversold reversal)
   - Fisher crosses below +1.5 from above → short (overbought reversal)
   - Proven to catch bear market rally reversals

3) 1d HMA(21) for macro trend filter:
   - Only long if price > 1d HMA (bullish bias)
   - Only short if price < 1d HMA (bearish bias)
   - Prevents counter-trend trades in strong moves

4) ATR(14) trailing stop at 2.5x — protects capital during 2022-style crashes

5) Position sizing: 0.25 base, 0.30 with regime confluence
   - Discrete levels to minimize fee churn

Why this should work on 12h:
- Regime detection adapts to market conditions (trend vs range)
- Fisher Transform excels in bear/range markets (2025 test period)
- 1d HTF filter prevents whipsaws on counter-trend trades
- 12h naturally produces 20-40 trades/year (low fee drag)
- Simpler than dual-regime strategies that failed (#119, #121, #122)

Target: 25-40 trades/year, Sharpe > 0.5 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_fisher_regime_1d_v1"
timeframe = "12h"
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
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2.0 * wma1 - wma2
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Calculate highest high and lowest low over period
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize to 0-1 range
    range_hl = highest - lowest
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    normalized = (hl2 - lowest) / range_hl
    
    # Clamp to avoid division issues
    normalized = np.clip(normalized, 0.001, 0.999)
    
    # Fisher transform
    fisher_input = 0.33 * 2.0 * (np.log(normalized / (1.0 - normalized)) + np.roll(np.log(normalized / (1.0 - normalized)), 1))
    fisher_input[0] = 0.0
    
    # Smooth Fisher
    fisher = pd.Series(fisher_input).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
    fisher = np.clip(fisher, -2.5, 2.5)  # Clamp extreme values
    
    return fisher

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = ranging market (mean reversion)
    CHOP < 38.2 = trending market (trend follow)
    """
    atr = calculate_atr(high, low, close, period)
    
    # Highest high and lowest low over period
    highest = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # CHOP formula
    range_hl = highest - lowest
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    atr_sum = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    chop = 100.0 * np.log10(atr_sum / range_hl) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (20-period high/low)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HMA for macro trend
    hma_1d = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    chop_14 = calculate_choppiness_index(high, low, close, period=14)
    fisher_9 = calculate_fisher_transform(high, low, period=9)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, period=20)
    
    signals = np.zeros(n)
    POSITION_SIZE_BASE = 0.25
    POSITION_SIZE_MAX = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(chop_14[i]) or np.isnan(fisher_9[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop_14[i] > 61.8  # Ranging market
        regime_trend = chop_14[i] < 38.2  # Trending market
        regime_neutral = not regime_chop and not regime_trend  # 38.2 <= CHOP <= 61.8
        
        # === FISHER TRANSFORM SIGNALS (for choppy regime) ===
        fisher_cross_long = False
        fisher_cross_short = False
        
        if i > 1 and not np.isnan(fisher_9[i-1]):
            # Long: Fisher crosses above -1.5 from below
            if fisher_9[i-1] < -1.5 and fisher_9[i] >= -1.5:
                fisher_cross_long = True
            # Short: Fisher crosses below +1.5 from above
            if fisher_9[i-1] > 1.5 and fisher_9[i] <= 1.5:
                fisher_cross_short = True
        
        # === DONCHIAN BREAKOUT SIGNALS (for trending regime) ===
        breakout_long = False
        breakout_short = False
        
        if i > 0 and not np.isnan(donchian_upper[i-1]) and not np.isnan(donchian_lower[i-1]):
            prev_high = donchian_upper[i-1]
            prev_low = donchian_lower[i-1]
            
            if close[i] > prev_high:
                breakout_long = True
            if close[i] < prev_low:
                breakout_short = True
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        new_signal = 0.0
        
        # --- LONG ENTRY ---
        if price_above_hma_1d:  # HTF trend filter
            if regime_chop and fisher_cross_long:
                # Mean reversion in choppy market
                new_signal = POSITION_SIZE_BASE
                if fisher_9[i] < -1.0:  # Deeper oversold = stronger signal
                    new_signal = POSITION_SIZE_MAX
            
            elif regime_trend and breakout_long:
                # Trend following in trending market
                new_signal = POSITION_SIZE_BASE
                if chop_14[i] < 30.0:  # Strong trend
                    new_signal = POSITION_SIZE_MAX
        
        # --- SHORT ENTRY ---
        if price_below_hma_1d:  # HTF trend filter
            if regime_chop and fisher_cross_short:
                # Mean reversion in choppy market
                new_signal = -POSITION_SIZE_BASE
                if fisher_9[i] > 1.0:  # Deeper overbought = stronger signal
                    new_signal = -POSITION_SIZE_MAX
            
            elif regime_trend and breakout_short:
                # Trend following in trending market
                new_signal = -POSITION_SIZE_BASE
                if chop_14[i] < 30.0:  # Strong trend
                    new_signal = -POSITION_SIZE_MAX
        
        # === HOLD POSITION LOGIC ===
        # Hold if regime hasn't changed dramatically and HTF trend intact
        if in_position and new_signal == 0.0:
            if position_side > 0:
                # Hold long if price > Donchian mid and HTF bullish
                if close[i] > donchian_mid[i] and price_above_hma_1d:
                    # Don't exit in choppy regime if Fisher not extreme
                    if not (regime_chop and fisher_9[i] > 1.0):
                        new_signal = signals[i-1] if i > 0 else 0.0
            elif position_side < 0:
                # Hold short if price < Donchian mid and HTF bearish
                if close[i] < donchian_mid[i] and price_below_hma_1d:
                    # Don't exit in choppy regime if Fisher not extreme
                    if not (regime_chop and fisher_9[i] < -1.0):
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
        
        # === EXIT ON HTF TREND REVERSAL ===
        if in_position and position_side > 0:
            if price_below_hma_1d:
                new_signal = 0.0
        
        if in_position and position_side < 0:
            if price_above_hma_1d:
                new_signal = 0.0
        
        # === EXIT ON OPPOSITE REGIME SIGNAL ===
        if in_position and position_side > 0 and regime_chop:
            # Exit long if Fisher becomes overbought in choppy market
            if fisher_9[i] > 1.5:
                new_signal = 0.0
        
        if in_position and position_side < 0 and regime_chop:
            # Exit short if Fisher becomes oversold in choppy market
            if fisher_9[i] < -1.5:
                new_signal = 0.0
        
        # === EXIT ON OPPOSITE DONCHIAN BREAK (trending regime) ===
        if in_position and position_side > 0 and regime_trend:
            if breakout_short:
                new_signal = 0.0
        
        if in_position and position_side < 0 and regime_trend:
            if breakout_long:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals