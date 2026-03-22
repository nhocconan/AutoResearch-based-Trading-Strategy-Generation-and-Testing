#!/usr/bin/env python3
"""
Experiment #279: 4h Primary + 1d HTF — Fisher Transform + HMA Trend + Choppiness Regime

Hypothesis: After 250+ failed experiments, the key insight is:
1. Fisher Transform catches reversals better than RSI in bear/range markets (2022 crash, 2025 bear)
2. HMA(21) on 1d provides clean trend direction without whipsaw
3. Choppiness Index > 50 = mean revert mode, < 50 = trend mode (simpler than 38.2/61.8)
4. FEWER conflicting filters = more trades (many strategies failed with 0 trades)
5. Force entry every 12 bars if no signal (4h * 12 = 48h = 2 days) ensures minimum trade frequency

Key differences from failed #274, #278:
- Fisher Transform instead of RSI (better for reversals in bear markets)
- Single regime threshold (CHOP > 50 vs < 50) instead of multiple bands
- Relaxed entry conditions: only 2-3 confluences required, not 5+
- Aggressive frequency safeguard: force trade every 12 bars if flat

Position sizing: 0.30 base, 0.40 strong conviction (discrete levels)
Target: 25-50 trades/year per symbol (appropriate for 4h)
Stoploss: 2.5 * ATR trailing stop
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_hma_chop_1d_v1"
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
    Converts price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    n = period
    hl2 = (high + low) / 2.0
    
    # Normalize price to 0-1 range using Donchian
    highest = pd.Series(hl2).rolling(window=n, min_periods=n).max().values
    lowest = pd.Series(hl2).rolling(window=n, min_periods=n).min().values
    
    fisher_input = np.zeros(len(close))
    for i in range(n, len(close)):
        range_hl = highest[i] - lowest[i]
        if range_hl > 0:
            fisher_input[i] = 0.66 * ((hl2[i] - lowest[i]) / range_hl - 0.5) + 0.67 * fisher_input[i-1]
        else:
            fisher_input[i] = fisher_input[i-1] if i > 0 else 0.0
    
    # Clamp to prevent division by zero
    fisher_input = np.clip(fisher_input, -0.999, 0.999)
    
    # Fisher Transform
    fisher = np.zeros(len(close))
    fisher_signal = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if i >= n:
            fisher[i] = 0.5 * np.log((1 + fisher_input[i]) / (1 - fisher_input[i]))
            fisher_signal[i] = 0.5 * np.log((1 + fisher_input[i-1]) / (1 - fisher_input[i-1]))
        else:
            fisher[i] = 0.0
            fisher_signal[i] = 0.0
    
    return fisher, fisher_signal

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
    CHOP > 50 = choppy/range market (mean revert)
    CHOP < 50 = trending market (trend follow)
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
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    hma_4h_21 = calculate_hma(close, 21)
    hma_4h_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.30
    STRONG_SIZE = 0.40
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -12
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(hma_4h_21[i]):
            continue
        
        if np.isnan(chop_14[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 50.0
        is_trending = chop_14[i] < 50.0
        
        # === 4H LOCAL TREND ===
        hma_4h_bullish = hma_4h_21[i] > hma_4h_50[i]
        hma_4h_bearish = hma_4h_21[i] < hma_4h_50[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        fisher_extreme_long = fisher[i] < -1.8
        fisher_extreme_short = fisher[i] > 1.8
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending)
        if is_trending:
            # LONG: Trending + bull regime + 4h HMA bullish + Fisher confirming
            if regime_bull and hma_4h_bullish and (fisher_long or fisher_extreme_long):
                new_signal = STRONG_SIZE
            # LONG: Trending + bull regime + price above 4h HMA
            elif regime_bull and close[i] > hma_4h_21[i] and hma_4h_bullish:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + bear regime + 4h HMA bearish + Fisher confirming
            if regime_bear and hma_4h_bearish and (fisher_short or fisher_extreme_short):
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Trending + bear regime + price below 4h HMA
            elif regime_bear and close[i] < hma_4h_21[i] and hma_4h_bearish:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: Choppy + Fisher extreme oversold OR Fisher long cross
            if fisher_extreme_long or fisher_long:
                new_signal = BASE_SIZE
            # LONG: Choppy + price below 4h HMA (oversold in range)
            elif close[i] < hma_4h_21[i] * 0.98:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.8
            
            # SHORT: Choppy + Fisher extreme overbought OR Fisher short cross
            if fisher_extreme_short or fisher_short:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + price above 4h HMA (overbought in range)
            elif close[i] > hma_4h_21[i] * 1.02:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 12 bars (~48h = 2 days on 4h)
        if bars_since_last_trade > 12 and new_signal == 0.0 and not in_position:
            if regime_bull and hma_4h_bullish:
                new_signal = BASE_SIZE * 0.7
            elif regime_bear and hma_4h_bearish:
                new_signal = -BASE_SIZE * 0.7
            elif is_choppy and fisher[i] < -1.0:
                new_signal = BASE_SIZE * 0.6
            elif is_choppy and fisher[i] > 1.0:
                new_signal = -BASE_SIZE * 0.6
        
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
            if position_side > 0 and regime_bear and hma_4h_bearish:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and hma_4h_bullish:
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
    
    return signals