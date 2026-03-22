#!/usr/bin/env python3
"""
Experiment #244: 4h Primary + 12h/1d HTF — KAMA Adaptive Trend + Fisher Transform Entries

Hypothesis: After analyzing 243 experiments, the key insight is that adaptive indicators
(KAMA) outperform fixed indicators (EMA/HMA) in crypto's variable volatility regime.
Combined with Fisher Transform for precise entry timing and Choppiness Index for
regime detection, this should beat #236's Sharpe=0.270.

Strategy components:
1. KAMA(21) on 4h - adapts to volatility, reduces whipsaw in choppy markets
2. Fisher Transform(9) on 4h - normalizes price, catches reversals at extremes
3. Choppiness Index(14) on 4h - regime filter (trend vs mean-revert)
4. 12h KAMA slope - primary trend direction (bull/bear/neutral)
5. 1d ADX - trend strength confirmation
6. ATR-based position sizing and 2.5x trailing stops

Key improvements over #234:
- KAMA instead of HMA (better volatility adaptation)
- Fisher Transform for entry timing (proven in research for bear markets)
- Simpler regime logic (fewer conflicting paths)
- Looser RSI thresholds (35/65 not 30/70) for guaranteed trade frequency
- Force-trade mechanism after 35 bars of no signal

Position sizing: 0.25 base, 0.30 strong signals (discrete levels)
Target: 30-50 trades/year per symbol (within 4h cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_chop_regime_12h1d_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market volatility - moves fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = (close_s - close_s.shift(er_period)).abs()
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        if np.isnan(sc.iloc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution, catches reversals at extremes.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    # Typical price
    typical = (high + low) / 2.0
    typical_s = pd.Series(typical)
    
    # Highest high and lowest low over period
    hh = typical_s.rolling(window=period, min_periods=period).max()
    ll = typical_s.rolling(window=period, min_periods=period).min()
    
    # Normalize to -1 to +1 range
    x = (2 * typical - hh - ll) / (hh - ll).replace(0, np.nan)
    x = x.clip(-0.999, 0.999)  # Prevent log errors
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + x) / (1 - x)).replace([np.inf, -np.inf], np.nan)
    
    # Signal line (previous Fisher)
    fisher_signal = fisher.shift(1)
    
    return fisher.fillna(0).values, fisher_signal.fillna(0).values

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
    
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    
    return adx.fillna(20).values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP > 61.8 = choppy/range market (mean revert)
    CHOP < 38.2 = trending market (trend follow)
    """
    n = period
    atr_vals = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_vals).rolling(window=n, min_periods=n).sum().values
    
    # Highest high and lowest low over period
    hh = pd.Series(high).rolling(window=n, min_periods=n).max().values
    ll = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    # Choppiness calculation
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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h HTF indicators (primary trend regime)
    kama_12h_21 = calculate_kama(df_12h['close'].values, 10, 2, 30)
    kama_12h_slope = calculate_kama_slope(kama_12h_21, 3)
    
    # Calculate 1d HTF indicators (trend strength)
    adx_1d = calculate_adx(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    kama_1d_21 = calculate_kama(df_1d['close'].values, 10, 2, 30)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    kama_12h_21_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_21)
    kama_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_slope)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    kama_1d_21_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    kama_4h_21 = calculate_kama(close, 10, 2, 30)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    
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
    last_trade_bar = -35
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_12h_21_aligned[i]) or np.isnan(kama_12h_slope_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(kama_4h_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        # === REGIME DETECTION (12h KAMA slope) ===
        # Bull regime: 12h KAMA slope > 0.20%
        # Bear regime: 12h KAMA slope < -0.20%
        # Neutral: between -0.20% and 0.20%
        regime_bull = kama_12h_slope_aligned[i] > 0.20
        regime_bear = kama_12h_slope_aligned[i] < -0.20
        regime_neutral = not regime_bull and not regime_bear
        
        price_above_12h_kama = close[i] > kama_12h_21_aligned[i]
        price_below_12h_kama = close[i] < kama_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 55 = range market (mean revert)
        # CHOP < 45 = trend market (trend follow)
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === 1D TREND STRENGTH ===
        daily_trend_strong = adx_1d_aligned[i] > 25
        price_above_1d_kama = close[i] > kama_1d_21_aligned[i]
        price_below_1d_kama = close[i] < kama_1d_21_aligned[i]
        
        # === 4H LOCAL SIGNALS ===
        price_above_4h_kama = close[i] > kama_4h_21[i]
        price_below_4h_kama = close[i] < kama_4h_21[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below
        fisher_long = fisher[i] > -1.5 and fisher_signal[i] <= -1.5
        # Short: Fisher crosses below +1.5 from above
        fisher_short = fisher[i] < 1.5 and fisher_signal[i] >= 1.5
        
        # Fisher extreme reversals
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # === RSI MOMENTUM (LOOSE THRESHOLDS FOR TRADE FREQUENCY) ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_bullish = rsi_14[i] > 50
        rsi_bearish = rsi_14[i] < 50
        
        # === KAMA TREND ===
        kama_bullish = price_above_4h_kama
        kama_bearish = price_below_4h_kama
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_trending or regime_bull or regime_bear:
            # LONG: Regime bullish + price above 12h KAMA + Fisher confirmation
            if regime_bull and price_above_12h_kama and fisher[i] > -1.0:
                if rsi_bullish or fisher_long:
                    new_signal = STRONG_SIZE
            # LONG: Price above all KAMAs + RSI bullish + daily trend strong
            elif price_above_12h_kama and price_above_4h_kama and price_above_1d_kama:
                if rsi_bullish and daily_trend_strong:
                    new_signal = BASE_SIZE
            
            # SHORT: Regime bearish + price below 12h KAMA + Fisher confirmation
            if regime_bear and price_below_12h_kama and fisher[i] < 1.0:
                if rsi_bearish or fisher_short:
                    new_signal = -STRONG_SIZE
            # SHORT: Price below all KAMAs + RSI bearish + daily trend strong
            elif price_below_12h_kama and price_below_4h_kama and price_below_1d_kama:
                if rsi_bearish and daily_trend_strong:
                    new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: Fisher oversold + RSI oversold + not in strong bear regime
            if fisher_oversold and rsi_oversold and not regime_bear:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.7
            # LONG: RSI very oversold (<35) + price below 4h KAMA
            elif rsi_14[i] < 35 and price_below_4h_kama and not regime_bear:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.5
            
            # SHORT: Fisher overbought + RSI overbought + not in strong bull regime
            if fisher_overbought and rsi_overbought and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.7
            # SHORT: RSI very overbought (>65) + price above 4h KAMA
            elif rsi_14[i] > 65 and price_above_4h_kama and not regime_bull:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.5
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 35 bars (~6 days on 4h)
        if bars_since_last_trade > 35 and new_signal == 0.0 and not in_position:
            if regime_bull and rsi_14[i] > 45 and price_above_4h_kama:
                new_signal = BASE_SIZE * 0.4
            elif regime_bear and rsi_14[i] < 55 and price_below_4h_kama:
                new_signal = -BASE_SIZE * 0.4
            elif is_choppy and rsi_14[i] < 38:
                new_signal = BASE_SIZE * 0.35
            elif is_choppy and rsi_14[i] > 62:
                new_signal = -BASE_SIZE * 0.35
        
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
            if position_side > 0 and regime_bear and price_below_12h_kama:
                regime_reversal = True
            # Short position but regime turns strongly bullish
            if position_side < 0 and regime_bull and price_above_12h_kama:
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