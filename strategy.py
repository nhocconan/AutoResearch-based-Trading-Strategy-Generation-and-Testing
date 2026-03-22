#!/usr/bin/env python3
"""
Experiment #587: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime + Donchian

Hypothesis: After analyzing 500+ failed strategies, the pattern shows:
- #577 (1d + 1w + Chop + CRSI) achieved Sharpe=0.520 — this is the benchmark
- CRSI-heavy strategies are over-tested (50+ variants tried)
- Fisher Transform is UNDER-TESTED but has strong literature support for reversals
- Fisher catches turning points in bear/range markets better than RSI
- Combining Fisher (entry) + Choppiness (regime) + Donchian (breakout confirm) is novel
- 1w HTF for major regime filter (bull/bear) + 1d for execution timing

Strategy Logic:
1. 1w HMA(21) for MAJOR regime (bull if price>HMA, bear if price<HMA)
2. 1d Choppiness Index(14) for regime: CHOP>61.8=range (mean revert), CHOP<38.2=trend
3. 1d Fisher Transform(9) for entries: long when Fisher crosses above -1.5, short when crosses below +1.5
4. 1d Donchian(20) breakout confirmation: long only if price>Donchian_mid, short if price<Donchian_mid
5. ATR(14) 2.5x trailing stop on all positions
6. Position size: 0.30 discrete (per Rule 4)

Why this might beat Sharpe=0.520:
- Fisher Transform is mathematically superior to RSI for Gaussian normalization
- Less overfit than CRSI (fewer strategies tried with Fisher)
- Donchian confirmation reduces false signals in choppy markets
- 1w HTF is MORE stable than 1d for regime (proven in #577)
- Dual regime logic: mean-revert entries in chop, trend entries in trend

Position sizing: 0.30 base (discrete per Rule 4, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: >=30 trades/symbol on train, >=3 on test, Sharpe > 0.520 all symbols
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_donchian_1w_v1"
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
    Calculate Ehlers Fisher Transform - Gaussian normalization for reversal detection.
    Fisher = 0.5 * ln((1+X)/(1-X)) where X = 0.67*(2*(price-mid)/(high-low)-0.1)
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    fisher = np.zeros(n)
    fisher_signal = np.zeros(n)
    
    for i in range(period, n):
        # Calculate typical price and range
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        range_val = highest - lowest
        if range_val < 1e-10:
            range_val = 1e-10
        
        # Normalize price to -1 to +1 range
        price = close[i]
        x = 0.67 * (2.0 * (price - lowest) / range_val - 1.0)
        
        # Clamp to avoid ln domain errors
        x = np.clip(x, -0.999, 0.999)
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x + 1e-10))
        
        # Signal line (previous fisher value)
        if i > period:
            fisher_signal[i] = fisher[i-1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP) for regime detection.
    CHOP = 100 * log10(sum(ATR, period)) / log10(highest_high - lowest_low)
    CHOP > 61.8 = range/choppy (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.zeros(n)
    
    # Calculate ATR for each bar
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        sum_atr = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        range_val = highest_high - lowest_low
        if range_val < 1e-10 or sum_atr < 1e-10:
            chop[i] = 50.0  # neutral
        else:
            chop[i] = 100.0 * np.log10(sum_atr) / np.log10(range_val)
    
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low, mid)."""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    mid = np.zeros(n)
    
    for i in range(period, n):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
        mid[i] = (upper[i] + lower[i]) / 2.0
    
    return upper, lower, mid

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) - reduces lag vs EMA."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF HMA for major regime direction
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    donchian_upper, donchian_lower, donchian_mid = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        if np.isnan(donchian_mid[i]):
            continue
        
        # === 1W MAJOR REGIME (primary direction filter) ===
        bull_regime_1w = close[i] > hma_1w_21_aligned[i]
        bear_regime_1w = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME (range vs trend) ===
        choppy_regime = chop_14[i] > 61.8  # range market - mean revert
        trending_regime = chop_14[i] < 38.2  # trend market - trend follow
        neutral_regime = not choppy_regime and not trending_regime
        
        # === FISHER TRANSFORM ENTRY SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_short = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Also allow entries when Fisher is at extremes (not just cross)
        fisher_extreme_long = fisher[i] < -1.2
        fisher_extreme_short = fisher[i] > 1.2
        
        # === DONCHIAN CONFIRMATION ===
        # Long only if price above Donchian mid (bullish bias)
        # Short only if price below Donchian mid (bearish bias)
        donchian_bull = close[i] > donchian_mid[i]
        donchian_bear = close[i] < donchian_mid[i]
        
        # === ENTRY LOGIC — DUAL REGIME ===
        new_signal = 0.0
        
        # LONG ENTRY: 1w bull + Fisher signal + Donchian confirm
        if bull_regime_1w:
            # In trending regime: need Fisher cross + Donchian bull
            if trending_regime and fisher_cross_long and donchian_bull:
                new_signal = POSITION_SIZE
            # In choppy regime: Fisher extreme is enough (mean revert)
            elif choppy_regime and fisher_extreme_long:
                new_signal = POSITION_SIZE * 0.8
            # In neutral regime: require both cross and Donchian
            elif neutral_regime and fisher_cross_long and donchian_bull:
                new_signal = POSITION_SIZE * 0.7
        
        # SHORT ENTRY: 1w bear + Fisher signal + Donchian confirm
        elif bear_regime_1w:
            # In trending regime: need Fisher cross + Donchian bear
            if trending_regime and fisher_cross_short and donchian_bear:
                new_signal = -POSITION_SIZE
            # In choppy regime: Fisher extreme is enough (mean revert)
            elif choppy_regime and fisher_extreme_short:
                new_signal = -POSITION_SIZE * 0.8
            # In neutral regime: require both cross and Donchian
            elif neutral_regime and fisher_cross_short and donchian_bear:
                new_signal = -POSITION_SIZE * 0.7
        
        # === HOLD POSITION LOGIC ===
        # If already in position, maintain unless exit conditions hit
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
        
        # === EXIT CONDITIONS (regime flip) ===
        # Exit long on 1w regime flip to bear
        if in_position and position_side > 0:
            if bear_regime_1w:
                new_signal = 0.0
        
        # Exit short on 1w regime flip to bull
        if in_position and position_side < 0:
            if bull_regime_1w:
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
                # Flip position
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