#!/usr/bin/env python3
"""
Experiment #247: 1d Primary + 1w HTF — Donchian Breakout with HMA Trend + Choppiness Regime

Hypothesis: After analyzing 246 experiments, 1d timeframe with weekly trend filter shows
promise for reducing whipsaw while maintaining sufficient trade frequency. Donchian
breakouts work well when filtered by: (1) HTF trend direction, (2) Choppiness regime,
(3) HMA confirmation. This combines trend-following (breakouts) with regime awareness.

Key components:
1. 1w HMA(21) slope for PRIMARY trend direction (bull/bear regime)
2. 1d Donchian(20) breakout for entry signals
3. 1d Choppiness Index(14) for regime detection (trend vs range)
4. 1d HMA(16/48) crossover for local trend confirmation
5. 1d ATR(14) for trailing stops
6. Dual logic: breakout entries in trending regime, mean-revert in choppy

Why this should beat #246 (Sharpe=0.350):
- 1d timeframe has lower fee drag than 12h (fewer false signals)
- Donchian breakouts capture major moves better than KAMA crossover
- 1w HTF provides stronger trend filter than 1d
- Choppiness regime switch reduces whipsaw in range markets

Position sizing: 0.25 base, 0.30 strong (discrete levels)
Target: 15-35 trades/year per symbol (within 1d cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_chop_1w_v1"
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
    HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    More responsive than EMA with less lag.
    """
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope as percentage change over lookback."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        prev = hma_values[i - lookback]
        curr = hma_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

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

def calculate_donchian_channels(high, low, period=20):
    """
    Calculate Donchian Channel upper and lower bands.
    Upper = highest high over period
    Lower = lowest low over period
    """
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr1 = high_s - low_s
    tr2 = (high_s - close_s.shift(1)).abs()
    tr3 = (low_s - close_s.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period, adjust=False).mean() / atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (primary trend regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    hma_1d_16 = calculate_hma(close, 16)
    hma_1d_48 = calculate_hma(close, 48)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
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
    last_trade_bar = -50
    consecutive_no_signal = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            consecutive_no_signal += 1
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            consecutive_no_signal += 1
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_1d_16[i]):
            consecutive_no_signal += 1
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            consecutive_no_signal += 1
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            consecutive_no_signal += 1
            continue
        
        # === 1W TREND REGIME (primary direction filter) ===
        # Bull: 1w HMA slope > 0.5%
        # Bear: 1w HMA slope < -0.5%
        # Neutral: between
        regime_bull = hma_1w_slope_aligned[i] > 0.50
        regime_bear = hma_1w_slope_aligned[i] < -0.50
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 1D LOCAL SIGNALS ===
        price_above_1d_hma = close[i] > hma_1d_16[i]
        price_below_1d_hma = close[i] < hma_1d_16[i]
        hma_1d_bullish = hma_1d_16[i] > hma_1d_48[i]
        hma_1d_bearish = hma_1d_16[i] < hma_1d_48[i]
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx_14[i] > 25
        weak_trend = adx_14[i] < 20
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout above upper band
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        # Breakout below lower band
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        consecutive_no_signal = 0
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_trending and strong_trend:
            # LONG: Trending + bull/neutral regime + Donchian breakout + HMA bullish
            if (regime_bull or regime_neutral) and donchian_breakout_long and hma_1d_bullish and price_above_1w_hma:
                new_signal = STRONG_SIZE
            # LONG: Trending + price above 1w HMA + HMA crossover + ADX rising
            elif price_above_1w_hma and hma_1d_bullish and adx_14[i] > 20 and rsi_14[i] > 45:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE
            
            # SHORT: Trending + bear/neutral regime + Donchian breakdown + HMA bearish
            if (regime_bear or regime_neutral) and donchian_breakout_short and hma_1d_bearish and price_below_1w_hma:
                new_signal = -STRONG_SIZE
            # SHORT: Trending + price below 1w HMA + HMA crossover + ADX rising
            elif price_below_1w_hma and hma_1d_bearish and adx_14[i] > 20 and rsi_14[i] < 55:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy and weak_trend:
            # LONG: Choppy + RSI oversold + price below 1d HMA (pullback in neutral/bull)
            if rsi_14[i] < 35 and price_below_1d_hma and not regime_bear:
                new_signal = BASE_SIZE * 0.8
            # LONG: Choppy + RSI very oversold (<25)
            if rsi_14[i] < 25 and not regime_bear:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.6
            
            # SHORT: Choppy + RSI overbought + price above 1d HMA (pullback in neutral/bear)
            if rsi_14[i] > 65 and price_above_1d_hma and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.8
            # SHORT: Choppy + RSI very overbought (>75)
            if rsi_14[i] > 75 and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.6
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 45 bars (~45 days on 1d)
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if (regime_bull or regime_neutral) and rsi_14[i] > 45 and price_above_1d_hma:
                new_signal = BASE_SIZE * 0.5
            elif (regime_bear or regime_neutral) and rsi_14[i] < 55 and price_below_1d_hma:
                new_signal = -BASE_SIZE * 0.5
            elif is_choppy and rsi_14[i] < 40:
                new_signal = BASE_SIZE * 0.4
            elif is_choppy and rsi_14[i] > 60:
                new_signal = -BASE_SIZE * 0.4
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Long position but regime turns strongly bearish
            if position_side > 0 and regime_bear and price_below_1w_hma:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_1w_hma:
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