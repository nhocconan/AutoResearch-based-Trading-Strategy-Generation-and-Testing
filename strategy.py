#!/usr/bin/env python3
"""
Experiment #291: 4h Primary + 1d/1w HTF — KAMA Adaptive Trend + RSI Z-Score + BB Regime

Hypothesis: After 12 consecutive 4h failures (#279-#290), simplify dramatically:
1. KAMA (Kaufman Adaptive) instead of HMA — adapts to volatility, less whipsaw
2. 1w HTF for PRIMARY regime (bull/bear) — stronger than 1d alone
3. RSI Z-Score (not raw RSI) — normalizes across regimes, better mean-reversion
4. Bollinger Band Width for volatility regime (squeeze = breakout potential)
5. Asymmetric sizing: smaller in chop, larger in strong trend
6. Force minimum 35 trades/year with relaxed RSI thresholds

Key differences from failed #284 (Connors):
- KAMA instead of HMA (adapts to market efficiency ratio)
- RSI Z-Score instead of CRSI (simpler, more robust)
- 1w HTF added for stronger regime filter
- BB Width for volatility regime (not just Choppiness)
- Simpler entry logic (fewer conflicting conditions)

Position sizing: 0.25 base, 0.35 strong (discrete)
Target: 35-55 trades/year on 4h
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_rsi_zscore_bb_1d1w_v1"
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

def calculate_rsi_zscore(close, rsi_period=14, zscore_period=100):
    """Calculate RSI Z-Score for normalized mean-reversion signals."""
    rsi = calculate_rsi(close, rsi_period)
    rsi_s = pd.Series(rsi)
    rsi_mean = rsi_s.rolling(window=zscore_period, min_periods=zscore_period).mean()
    rsi_std = rsi_s.rolling(window=zscore_period, min_periods=zscore_period).std()
    rsi_z = (rsi - rsi_mean) / rsi_std.replace(0, np.nan)
    return rsi_z.fillna(0).values

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market noise via Efficiency Ratio.
    ER = |close - close_n| / sum(|close_i - close_i-1|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    KAMA = KAMA_prev + SC * (close - KAMA_prev)
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Efficiency Ratio numerator: absolute price change over er_period
    er_num = np.abs(close - np.roll(close, er_period))
    er_num[:er_period] = np.nan
    
    # Efficiency Ratio denominator: sum of absolute changes
    er_den = np.zeros(n)
    for i in range(er_period, n):
        er_den[i] = np.sum(np.abs(close[i-er_period+1:i+1] - np.roll(close[i-er_period+1:i+1], 1)))
    
    er = er_num / np.where(er_den > 0, er_den, np.nan)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # Initialize KAMA
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands and Band Width."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    width = (upper - lower) / sma * 100  # Band Width as percentage
    
    return upper.values, lower.values, width.values

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA) for HTF trend."""
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate 1w HTF indicators (primary regime)
    hma_1w_10 = calculate_hma(df_1w['close'].values, 10)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_10_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_10)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_4h_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_4h_30 = calculate_kama(close, er_period=10, fast_period=2, slow_period=60)
    rsi_z_14 = calculate_rsi_zscore(close, rsi_period=14, zscore_period=100)
    rsi_14 = calculate_rsi(close, 14)
    bb_upper, bb_lower, bb_width = calculate_bollinger_bands(close, 20, 2.0)
    
    # Calculate BB Width percentile for squeeze detection
    bb_width_s = pd.Series(bb_width)
    bb_width_pct = bb_width_s.rolling(window=100, min_periods=100).rank(pct=True).values
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.35
    WEAK_SIZE = 0.15
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -15
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_10_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_4h_10[i]) or np.isnan(rsi_z_14[i]):
            continue
        
        if np.isnan(bb_width[i]) or np.isnan(bb_width_pct[i]):
            continue
        
        # === 1W PRIMARY REGIME (strongest filter) ===
        # Bull: price above 1w HMA(10)
        # Bear: price below 1w HMA(10)
        regime_1w_bull = close[i] > hma_1w_10_aligned[i]
        regime_1w_bear = close[i] < hma_1w_10_aligned[i]
        
        # === 1D INTERMEDIATE TREND ===
        trend_1d_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        trend_1d_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === 4H KAMA TREND ===
        kama_bullish = kama_4h_10[i] > kama_4h_30[i]
        kama_bearish = kama_4h_10[i] < kama_4h_30[i]
        kama_slope_up = kama_4h_10[i] > kama_4h_10[i-5] if i >= 5 else False
        kama_slope_down = kama_4h_10[i] < kama_4h_10[i-5] if i >= 5 else False
        
        # === RSI Z-SCORE (mean reversion signal) ===
        # Z < -1.5 = oversold, Z > +1.5 = overbought
        rsi_z_oversold = rsi_z_14[i] < -1.2
        rsi_z_overbought = rsi_z_14[i] > 1.2
        rsi_z_extreme_oversold = rsi_z_14[i] < -2.0
        rsi_z_extreme_overbought = rsi_z_14[i] > 2.0
        
        # === BOLLINGER BAND REGIME ===
        # BB Width < 20th percentile = squeeze (breakout potential)
        # BB Width > 80th percentile = expansion (mean revert)
        bb_squeeze = bb_width_pct[i] < 0.25
        bb_expansion = bb_width_pct[i] > 0.75
        
        # === PRICE POSITION vs BB ===
        price_near_lower_bb = close[i] < bb_lower[i] * 1.005
        price_near_upper_bb = close[i] > bb_upper[i] * 0.995
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # CONFIDENCE SCORING
        long_confidence = 0
        short_confidence = 0
        
        # Long confidence factors
        if regime_1w_bull:
            long_confidence += 2
        if trend_1d_bull:
            long_confidence += 1
        if kama_bullish:
            long_confidence += 1
        if kama_slope_up:
            long_confidence += 1
        if rsi_z_oversold or price_near_lower_bb:
            long_confidence += 1
        if rsi_z_extreme_oversold:
            long_confidence += 2
        
        # Short confidence factors
        if regime_1w_bear:
            short_confidence += 2
        if trend_1d_bear:
            short_confidence += 1
        if kama_bearish:
            short_confidence += 1
        if kama_slope_down:
            short_confidence += 1
        if rsi_z_overbought or price_near_upper_bb:
            short_confidence += 1
        if rsi_z_extreme_overbought:
            short_confidence += 2
        
        # BB squeeze breakout logic
        if bb_squeeze:
            if kama_bullish and regime_1w_bull:
                long_confidence += 2
            if kama_bearish and regime_1w_bear:
                short_confidence += 2
        
        # ENTRY DECISION
        if long_confidence >= 4 and short_confidence < 3:
            if long_confidence >= 6:
                new_signal = STRONG_SIZE
            else:
                new_signal = BASE_SIZE
        elif short_confidence >= 4 and long_confidence < 3:
            if short_confidence >= 6:
                new_signal = -STRONG_SIZE
            else:
                new_signal = -BASE_SIZE
        elif long_confidence >= 3 and rsi_z_extreme_oversold:
            new_signal = BASE_SIZE
        elif short_confidence >= 3 and rsi_z_extreme_overbought:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 35+ trades/year) ===
        # Force trade if no signal for 15 bars (~60h = 2.5 days on 4h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if regime_1w_bull and kama_bullish and rsi_z_14[i] < -0.5:
                new_signal = WEAK_SIZE
            elif regime_1w_bear and kama_bearish and rsi_z_14[i] > 0.5:
                new_signal = -WEAK_SIZE
            elif bb_expansion and price_near_lower_bb:
                new_signal = WEAK_SIZE
            elif bb_expansion and price_near_upper_bb:
                new_signal = -WEAK_SIZE
        
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
            # Long position but 1w regime turns bearish + 4h KAMA bearish
            if position_side > 0 and regime_1w_bear and kama_bearish:
                regime_reversal = True
            # Short position but 1w regime turns bullish + 4h KAMA bullish
            if position_side < 0 and regime_1w_bull and kama_bullish:
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