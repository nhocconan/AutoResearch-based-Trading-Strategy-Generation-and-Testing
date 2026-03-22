#!/usr/bin/env python3
"""
Experiment #261: 4h Primary + 1d HTF — Simplified Regime Switch + HMA Trend + RSI Entries

Hypothesis: #259 failed (Sharpe=-0.034) due to too many conflicting conditions.
Simplify to proven components from #251 (Sharpe=0.155):
1. Choppiness Index ONLY for regime (trend vs mean-revert) - remove ADX redundancy
2. 1d HMA(21) as PRIMARY trend filter - only trade WITH daily trend
3. 4h RSI(14) for entry timing - simpler than CRSI, less overfitting
4. Remove Donchian breakout (too many false signals on 4h)
5. Cleaner position management - track state properly for stoploss
6. 2.0*ATR stoploss (tighter but with better filtered entries)
7. Force trade frequency by relaxing RSI thresholds slightly

Key changes from #259:
- Remove ADX (redundant with Choppiness)
- Remove Donchian (false breakouts on 4h)
- Remove frequency safeguard (let signals flow naturally)
- Simpler regime: CHOP>55=mean-revert, CHOP<45=trend, else flat
- Only trade WITH 1d HMA trend direction
- Discrete sizing: 0.0, ±0.25, ±0.30

Position sizing: 0.25 base, 0.30 strong conviction
Target: 25-50 trades/year (appropriate for 4h)
Stoploss: 2.0 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_simp_chop_hma_rsi_1d_v3"
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

def calculate_rsi(close, period=14):
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, window):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window, min_periods=window).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    chop = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = hh[i] - ll[i]
        if range_hl > 0 and atr_sum[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / range_hl) / np.log10(n)
        else:
            chop[i] = 50.0
    
    return chop

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (primary trend regime)
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_4h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_4h_21[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(chop_14[i]):
            signals[i] = 0.0
            continue
        
        # === 1D TREND REGIME (PRIMARY FILTER - only trade with daily trend) ===
        price_vs_1d_hma = close[i] - hma_1d_21_aligned[i]
        pct_vs_1d_hma = price_vs_1d_hma / hma_1d_21_aligned[i] if hma_1d_21_aligned[i] != 0 else 0
        
        # Strong bull: price > 1d HMA by >0.5%
        # Strong bear: price < 1d HMA by >0.5%
        # Neutral: within 0.5% of 1d HMA
        regime_bull_strong = pct_vs_1d_hma > 0.005
        regime_bear_strong = pct_vs_1d_hma < -0.005
        regime_neutral = not regime_bull_strong and not regime_bear_strong
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (trend follow entries)
        # 45-55 = transition (reduce size or flat)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        is_transition = not is_choppy and not is_trending
        
        # === 4H LOCAL SIGNALS ===
        price_vs_4h_hma = close[i] - hma_4h_21[i]
        pct_vs_4h_hma = price_vs_4h_hma / hma_4h_21[i] if hma_4h_21[i] != 0 else 0
        
        # RSI thresholds (relaxed for more trades)
        rsi_oversold = rsi_14[i] < 40.0
        rsi_overbought = rsi_14[i] > 60.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        
        # TREND FOLLOWING MODE (trending regime + aligned with 1d trend)
        if is_trending and not regime_neutral:
            # LONG: Trending + strong bull 1d + price above 4h HMA + RSI confirming momentum
            if regime_bull_strong and pct_vs_4h_hma > 0 and rsi_14[i] > 45:
                new_signal = STRONG_SIZE
            
            # SHORT: Trending + strong bear 1d + price below 4h HMA + RSI confirming momentum
            if regime_bear_strong and pct_vs_4h_hma < 0 and rsi_14[i] < 55:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
        
        # MEAN REVERSION MODE (choppy regime - trade against extremes)
        if is_choppy:
            # LONG: Choppy + RSI oversold + 1d not strongly bearish
            if rsi_oversold and not regime_bear_strong:
                new_signal = BASE_SIZE
            
            # LONG: Choppy + RSI extreme oversold (any 1d regime)
            if rsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Choppy + RSI overbought + 1d not strongly bullish
            if rsi_overbought and not regime_bull_strong:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            
            # SHORT: Choppy + RSI extreme overbought (any 1d regime)
            if rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # TRANSITION MODE (reduce size, only strong signals)
        if is_transition:
            # Only enter on extreme RSI with 1d trend confirmation
            if rsi_extreme_oversold and regime_bull_strong:
                new_signal = BASE_SIZE * 0.8
            if rsi_extreme_overbought and regime_bear_strong:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price since entry for long
                if close[i] > highest_since_entry:
                    highest_since_entry = close[i]
                # Trailing stop: highest - 2.0*ATR
                stoploss_price = highest_since_entry - 2.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price since entry for short
                if lowest_since_entry == 0.0 or close[i] < lowest_since_entry:
                    lowest_since_entry = close[i]
                # Trailing stop: lowest + 2.0*ATR
                stoploss_price = lowest_since_entry + 2.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        # Exit if 1d trend strongly reverses against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but 1d turns strongly bearish
            if position_side > 0 and regime_bear_strong:
                regime_reversal = True
            # Short position but 1d turns strongly bullish
            if position_side < 0 and regime_bull_strong:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        prev_signal = signals[i-1] if i > 0 else 0.0
        
        if new_signal != 0.0:
            if not in_position:
                # New position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            # Update tracking prices for existing position
            if position_side > 0 and close[i] > highest_since_entry:
                highest_since_entry = close[i]
            if position_side < 0 and (lowest_since_entry == 0.0 or close[i] < lowest_since_entry):
                lowest_since_entry = close[i]
        else:
            if in_position:
                # Position closed
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals