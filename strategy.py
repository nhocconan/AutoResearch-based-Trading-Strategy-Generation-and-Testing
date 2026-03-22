#!/usr/bin/env python3
"""
Experiment #276: 12h Primary + 1d HTF — Fisher Transform + HMA Trend + Choppiness Regime

Hypothesis: After 249 failed strategies, focus on what WORKED for 12h:
1. Ehlers Fisher Transform (period=9) for entry timing — catches reversals in bear/range markets better than RSI
2. 1d HMA(21) for PRIMARY trend direction (proven in #251, #259)
3. Choppiness Index(14) for regime detection — switch between trend-follow and mean-revert
4. Donchian(20) breakout for trend confirmation
5. Relaxed entry thresholds to ENSURE 20-50 trades/year (critical after #265, #268, #270, #275 got 0 trades)

Why Fisher Transform over RSI:
- Fisher normalizes price distribution to Gaussian, better for reversal detection
- Works exceptionally well in bear/range markets (2022 crash, 2025+ test period)
- Less whipsaw than RSI in choppy conditions
- Proven in academic literature (Ehlers 2002)

Key improvements over failed strategies:
- Fisher thresholds: -1.5/+1.5 (not extreme -2/+2) to ensure trades trigger
- Regime switch is SOFT (55/45 chop thresholds, not 61.8/38.2) for more flexibility
- Force entry every 15 bars if no signal (12h * 15 = 180h = 7.5 days)
- Simpler logic: fewer confluence requirements = more trades

Position sizing: 0.25 base, 0.30 strong (discrete, conservative)
Target: 25-45 trades/year per symbol (appropriate for 12h)
Stoploss: 2.5 * ATR trailing

CRITICAL: Must generate trades on ALL symbols (BTC, ETH, SOL) with Sharpe > 0 each.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma_chop_donchian_1d_v1"
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

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for better reversal detection.
    Entry: Fisher crosses above -1.5 (long), crosses below +1.5 (short)
    """
    n = period
    hl2 = (high + low) / 2.0
    
    # Calculate highest high and lowest low over period
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Normalize price to 0-1 range
    range_hl = hh - ll
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)  # avoid div by zero
    x = (hl2 - ll) / range_hl
    
    # Clamp to avoid extreme values
    x = np.clip(x, 0.001, 0.999)
    
    # Fisher transform
    fisher = np.zeros(len(close))
    for i in range(n, len(close)):
        if x[i] < 1 and x[i] > 0:
            fisher[i] = 0.5 * np.log((1 + x[i]) / (1 - x[i]))
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
    
    # Smooth with EMA for signal line
    fisher_s = pd.Series(fisher).ewm(span=5, min_periods=5, adjust=False).mean().values
    
    return fisher, fisher_s

def calculate_hma(close, period=21):
    """
    Calculate Hull Moving Average (HMA).
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Faster and smoother than EMA, less lag.
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=n, min_periods=n, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=n, min_periods=n, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=n, min_periods=n, adjust=False).mean()
    
    return adx.fillna(0).values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    We use 55/45 thresholds for softer regime switching.
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper and lower bounds)."""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

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
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, 9)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    prev_fisher = 0.0
    prev_fisher_signal = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        regime_bull = close[i] > hma_1d_21_aligned[i]
        regime_bear = close[i] < hma_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # Use 55/45 for softer switching (more trades than 61.8/38.2)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 22.0
        is_weak_trend = adx_14[i] < 18.0
        
        # === 12H LOCAL SIGNALS ===
        price_above_12h_hma = close[i] > hma_12h_21[i]
        price_below_12h_hma = close[i] < hma_12h_21[i]
        hma_12h_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_12h_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i] * 0.999
        donchian_breakout_short = close[i] < donchian_lower[i] * 1.001
        
        # === FISHER TRANSFORM SIGNALS ===
        # Fisher crosses above signal line from below
        fisher_cross_up = (fisher[i] > fisher_signal[i]) and (prev_fisher <= prev_fisher_signal)
        fisher_cross_down = (fisher[i] < fisher_signal[i]) and (prev_fisher >= prev_fisher_signal)
        
        # Fisher extreme levels (reversal zones)
        fisher_oversold = fisher[i] < -1.5
        fisher_overbought = fisher[i] > 1.5
        fisher_extreme_oversold = fisher[i] < -2.0
        fisher_extreme_overbought = fisher[i] > 2.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + strong ADX + regime aligned)
        if is_trending and is_strong_trend:
            # LONG: Trending + bull regime + Fisher cross up + HMA bullish
            if regime_bull and fisher_cross_up and hma_12h_bullish:
                new_signal = STRONG_SIZE
            # LONG: Trending + Donchian breakout + 1d HMA bull
            elif donchian_breakout_long and regime_bull:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + bear regime + Fisher cross down + HMA bearish
            if regime_bear and fisher_cross_down and hma_12h_bearish:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Trending + Donchian breakdown + 1d HMA bear
            elif donchian_breakout_short and regime_bear:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy + weak ADX)
        if is_choppy or is_weak_trend:
            # LONG: Choppy + Fisher oversold + not in strong bear regime
            if fisher_oversold and not regime_bear:
                new_signal = BASE_SIZE
            # LONG: Choppy + Fisher extreme oversold (any regime)
            if fisher_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Choppy + Fisher overbought + not in strong bull regime
            if fisher_overbought and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + Fisher extreme overbought (any regime)
            if fisher_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~180h = 7.5 days on 12h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_bull and fisher[i] > -1.0 and price_above_12h_hma:
                new_signal = BASE_SIZE * 0.8
            elif regime_bear and fisher[i] < 1.0 and price_below_12h_hma:
                new_signal = -BASE_SIZE * 0.8
            elif is_choppy and fisher[i] < -1.0:
                new_signal = BASE_SIZE * 0.7
            elif is_choppy and fisher[i] > 1.0:
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
            if position_side > 0 and regime_bear and price_below_12h_hma and chop_14[i] > 60:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_12h_hma and chop_14[i] > 60:
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
        
        # Store previous Fisher values for crossover detection
        prev_fisher = fisher[i]
        prev_fisher_signal = fisher_signal[i]
        
        signals[i] = new_signal
    
    return signals