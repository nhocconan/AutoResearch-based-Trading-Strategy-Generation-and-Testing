#!/usr/bin/env python3
"""
Experiment #256: 12h Primary + 1d HTF — Dual Regime (Trend/Mean-Revert) + Donchian Breakout

Hypothesis: After 255 experiments, the winning pattern combines:
1. 1d KAMA slope for PRIMARY trend direction (stable, less whipsaw than 12h)
2. 12h Choppiness Index for regime detection (trend vs mean-revert mode)
3. 12h Donchian(20) breakout for trend entries (proven on #246)
4. 12h RSI(14) extremes for mean-revert entries (wider thresholds for more trades)
5. ADX(14) filter to avoid weak signals
6. Asymmetric sizing: 0.25 base, 0.30 strong conviction
7. Minimum trade frequency enforcement (critical for 10+ trades/year)

Key insight from #246 (Sharpe=0.350): KAMA+Choppiness on 12h works.
Improvement: Use 1d for primary trend (more stable), 12h for entries.
Dual regime: CHOP>55 = mean revert (RSI extremes), CHOP<45 = trend (Donchian)

Position sizing: 0.25 base, 0.30 strong (discrete levels, max 0.35)
Target: 25-50 trades/year per symbol (~1 trade per week on 12h)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_donchian_rsi_1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise - fast in trends, slow in chop.
    """
    n = len(close)
    kama = np.zeros(n)
    kama[0] = close[0]
    
    for i in range(1, n):
        if i >= er_period:
            signal = abs(close[i] - close[i - er_period])
            noise = 0.0
            for j in range(i - er_period + 1, i + 1):
                noise += abs(close[j] - close[j - 1])
            er = signal / noise if noise > 0 else 0.0
        else:
            er = 0.0
        
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_kama_slope(kama_values, lookback=3):
    """Calculate KAMA slope as percentage change over lookback."""
    slope = np.zeros(len(kama_values))
    for i in range(lookback, len(kama_values)):
        prev = kama_values[i - lookback]
        curr = kama_values[i]
        if prev != 0 and not np.isnan(prev) and not np.isnan(curr):
            slope[i] = (curr - prev) / abs(prev) * 100
    return slope

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
    """Calculate Donchian Channel upper and lower bands."""
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
    kama_1d_21 = calculate_kama(df_1d['close'].values, 10, 2, 30)
    kama_1d_slope = calculate_kama_slope(kama_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    kama_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_slope)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    adx_14 = calculate_adx(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_12h_21 = calculate_kama(close, 10, 2, 30)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_1d_21_aligned[i]) or np.isnan(kama_1d_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_12h_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(adx_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === 1D TREND REGIME (primary direction filter) ===
        # Bull: 1d KAMA slope > 0.15%
        # Bear: 1d KAMA slope < -0.15%
        regime_bull = kama_1d_slope_aligned[i] > 0.15
        regime_bear = kama_1d_slope_aligned[i] < -0.15
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_1d_kama = close[i] > kama_1d_21_aligned[i]
        price_below_1d_kama = close[i] < kama_1d_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert entries)
        # CHOP < 45 = trend market (breakout entries)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === TREND STRENGTH ===
        is_strong_trend = adx_14[i] > 22.0
        is_weak_trend = adx_14[i] < 18.0
        
        # === 12H LOCAL SIGNALS ===
        price_above_kama = close[i] > kama_12h_21[i]
        price_below_kama = close[i] < kama_12h_21[i]
        
        # Donchian breakout signals
        breakout_long = close[i] > donchian_upper[i] * 0.999  # near upper band
        breakout_short = close[i] < donchian_lower[i] * 1.001  # near lower band
        
        # === RSI THRESHOLDS (wider for more trades) ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 25.0
        rsi_extreme_overbought = rsi_14[i] > 75.0
        rsi_mid_bull = rsi_14[i] > 45.0
        rsi_mid_bear = rsi_14[i] < 55.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_trending and is_strong_trend:
            # LONG: Trending + bull regime + Donchian breakout + RSI confirming
            if regime_bull and breakout_long and rsi_mid_bull:
                new_signal = STRONG_SIZE
            # LONG: Trending + price above 1d KAMA + price above 12h KAMA
            elif price_above_1d_kama and price_above_kama and rsi_14[i] > 40:
                new_signal = BASE_SIZE
            
            # SHORT: Trending + bear regime + Donchian breakout + RSI confirming
            if regime_bear and breakout_short and rsi_mid_bear:
                if new_signal == 0.0 or abs(new_signal) < STRONG_SIZE:
                    new_signal = -STRONG_SIZE
            # SHORT: Trending + price below 1d KAMA + price below 12h KAMA
            elif price_below_1d_kama and price_below_kama and rsi_14[i] < 60:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy + weak ADX)
        if is_choppy or is_weak_trend:
            # LONG: Choppy + RSI oversold (<35) + not in strong bear
            if rsi_oversold and not regime_bear:
                new_signal = BASE_SIZE
            # LONG: Choppy + RSI extreme oversold (<25) in any regime
            if rsi_extreme_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.9
            
            # SHORT: Choppy + RSI overbought (>65) + not in strong bull
            if rsi_overbought and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
            # SHORT: Choppy + RSI extreme overbought (>75) in any regime
            if rsi_extreme_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.9
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 20 bars (~10 days on 12h)
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 40 and price_above_kama:
                new_signal = BASE_SIZE * 0.7
            elif regime_bear and rsi_14[i] < 60 and price_below_kama:
                new_signal = -BASE_SIZE * 0.7
            elif is_choppy and rsi_14[i] < 40:
                new_signal = BASE_SIZE * 0.6
            elif is_choppy and rsi_14[i] > 60:
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
            if position_side > 0 and regime_bear and price_below_1d_kama:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_1d_kama:
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