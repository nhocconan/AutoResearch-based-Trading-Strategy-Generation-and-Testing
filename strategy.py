#!/usr/bin/env python3
"""
Experiment #269: 4h Primary + 1d HTF — Fisher Transform + Choppiness Regime + Vol Filter

Hypothesis: After 242 failed strategies with HMA/RSI/Chop combinations, try Fisher Transform
which excels at catching reversals in bear/range markets (2022 crash, 2025 bear).

Key differences from failed attempts:
1. Fisher Transform instead of RSI (better at extremes, less whipsaw)
2. Volatility filter: ATR ratio to avoid low-vol traps
3. Simpler regime: Just Choppiness (not ADX+Chop which overfits)
4. More aggressive entries: Fisher crosses at -1.5/+1.5 (not extreme 2.0)
5. Ensure trades: Force entry every 15 bars if no signal

Position sizing: 0.30 base (discrete), max 0.35
Target: 25-50 trades/year on 4h
Stoploss: 2.5 * ATR trailing

Why this might work:
- Fisher Transform normalizes price to Gaussian, catches turning points better than RSI
- Works well in bear markets (2022, 2025) where trend strategies fail
- Choppiness filter avoids trend-following in range markets
- 1d HMA provides primary trend bias without over-filtering
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_vol_1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution.
    Long when Fisher crosses above -1.5, short when crosses below +1.5.
    """
    n = period
    price = pd.Series((high + low) / 2)
    
    # Normalize price to range -1 to +1
    hh = price.rolling(window=n, min_periods=n).max()
    ll = price.rolling(window=n, min_periods=n).min()
    
    normalized = 2 * ((price - ll) / (hh - ll).replace(0, np.nan)) - 1
    normalized = normalized.clip(-0.999, 0.999)  # Avoid log(0)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized).replace(0, np.nan))
    fisher_prev = fisher.shift(1)
    
    return fisher.fillna(0).values, fisher_prev.fillna(0).values

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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
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

def calculate_atr_ratio(high, low, close, short_period=7, long_period=30):
    """Calculate ATR ratio for volatility spike detection."""
    atr_short = calculate_atr(high, low, close, short_period)
    atr_long = calculate_atr(high, low, close, long_period)
    
    ratio = np.zeros(len(close))
    for i in range(len(close)):
        if atr_long[i] > 0:
            ratio[i] = atr_short[i] / atr_long[i]
        else:
            ratio[i] = 1.0
    
    return ratio

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
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, 9)
    atr_ratio = calculate_atr_ratio(high, low, close, 7, 30)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    STRONG_SIZE = 0.35
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    prev_fisher_long = False
    prev_fisher_short = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(atr_ratio[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries with Fisher)
        # CHOP < 45 = trend market (trend entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === VOLATILITY FILTER ===
        # ATR ratio > 1.5 = vol spike (good for mean reversion)
        # ATR ratio < 0.8 = low vol (avoid entries)
        vol_spike = atr_ratio[i] > 1.5
        low_vol = atr_ratio[i] < 0.8
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long_cross = (fisher_prev[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short_cross = (fisher_prev[i] > 1.5) and (fisher[i] <= 1.5)
        
        # Strong Fisher signals (extreme reversals)
        fisher_strong_long = fisher[i] < -2.0
        fisher_strong_short = fisher[i] > 2.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MEAN REVERSION MODE (choppy market + Fisher extremes)
        if is_choppy:
            # Long: Choppy + Fisher oversold cross + not in strong bear regime
            if fisher_long_cross and not regime_bear:
                new_signal = BASE_SIZE
            # Long: Choppy + Fisher extreme oversold (any regime)
            if fisher_strong_long and vol_spike:
                if new_signal == 0.0:
                    new_signal = STRONG_SIZE
            
            # Short: Choppy + Fisher overbought cross + not in strong bull regime
            if fisher_short_cross and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # Short: Choppy + Fisher extreme overbought (any regime)
            if fisher_strong_short and vol_spike:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
        
        # TREND MODE (trending market + regime aligned)
        if is_trending:
            # Long: Trending + bull regime + Fisher turning up
            if regime_bull and fisher[i] > fisher_prev[i] and fisher[i] > -1.0:
                new_signal = BASE_SIZE
            # Short: Trending + bear regime + Fisher turning down
            if regime_bear and fisher[i] < fisher_prev[i] and fisher[i] < 1.0:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === VOL SPIKE REVERSION (works in any regime) ===
        if vol_spike and not low_vol:
            # Long: Vol spike + Fisher extreme oversold
            if fisher_strong_long:
                if new_signal == 0.0:
                    new_signal = STRONG_SIZE
            # Short: Vol spike + Fisher extreme overbought
            if fisher_strong_short:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~60h = 2.5 days on 4h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and fisher[i] > -1.0 and not low_vol:
                new_signal = BASE_SIZE * 0.8
            elif regime_bear and fisher[i] < 1.0 and not low_vol:
                new_signal = -BASE_SIZE * 0.8
            elif is_choppy and fisher_strong_long:
                new_signal = BASE_SIZE * 0.7
            elif is_choppy and fisher_strong_short:
                new_signal = -BASE_SIZE * 0.7
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and is_trending:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and is_trending:
                regime_reversal = True
        
        if stoploss_triggered or regime_reversal:
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
        prev_fisher_long = fisher_long_cross
        prev_fisher_short = fisher_short_cross
    
    return signals