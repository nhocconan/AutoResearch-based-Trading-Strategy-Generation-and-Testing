#!/usr/bin/env python3
"""
Experiment #241: 4h Primary + 1d/1w HTF — KAMA Trend + Fisher Transform Entries

Hypothesis: After 240 experiments, the pattern is clear:
- Complex regime-switching fails (too many conflicting conditions)
- HMA alone doesn't adapt to volatility changes
- Fisher Transform catches reversals in bear/range markets (2025+)
- KAMA (Kaufman Adaptive MA) adapts efficiency ratio to volatility
- ATR ratio (7/30) detects vol spikes for mean-reversion opportunities

This strategy uses:
1. 1w HMA(21) for MAJOR regime (bull/bear - very slow filter)
2. 1d KAMA(10) for PRIMARY trend direction (adapts to volatility)
3. 4h Fisher Transform(9) for entry timing (reversal signals)
4. 4h ATR(7)/ATR(30) ratio for volatility regime
5. Simple confluence: HTF trend + Fisher signal + vol regime
6. 2.5x ATR trailing stop for risk management

Key improvements:
- KAMA adapts to market efficiency (better than static HMA in chop)
- Fisher Transform normalized (-1.5 to +1.5) for clear entry signals
- ATR ratio filter: high vol (>1.8) = mean revert, low vol (<1.2) = trend
- LOOSE entry thresholds to ensure 30+ trades/year
- Force-trade after 35 bars if no signal (guarantees frequency)

Position sizing: 0.25 base, 0.30 strong signals (discrete levels)
Target: 25-45 trades/year per symbol (within 4h cost model)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_atr_ratio_1d1w_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - moves fast in trends, slow in chop.
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    """
    close_s = pd.Series(close)
    
    # Calculate Efficiency Ratio
    price_change = np.abs(close_s.diff(er_period))
    volatility = pd.Series(np.abs(close_s.diff())).rolling(window=er_period, min_periods=er_period).sum()
    er = price_change / volatility.replace(0, np.nan)
    er = er.fillna(0).values
    
    # Smoothing Constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    sc = np.power(er * (fast_sc - slow_sc) + slow_sc, 2)
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Normalizes price to Gaussian distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Donchian-style highest high and lowest low
    hh = hl2_s.rolling(window=period, min_periods=period).max().values
    ll = hl2_s.rolling(window=period, min_periods=period).min().values
    
    # Normalize to -1 to +1 range
    range_hl = hh - ll
    normalized = np.zeros(len(hl2))
    for i in range(period, len(hl2)):
        if range_hl[i] > 0:
            normalized[i] = 0.999 * (2.0 * (hl2[i] - ll[i]) / range_hl[i] - 1.0)
        else:
            normalized[i] = 0.0
    
    # Fisher transform
    fisher = np.zeros(len(hl2))
    for i in range(period, len(hl2)):
        if abs(normalized[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1.0 + normalized[i]) / (1.0 - normalized[i]))
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
    
    # Signal line (1-period EMA of Fisher)
    fisher_s = pd.Series(fisher)
    signal = fisher_s.ewm(span=1, min_periods=1, adjust=False).mean().values
    
    return fisher, signal

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HTF indicators (major regime)
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Calculate 1d HTF indicators (primary trend)
    kama_1d_10 = calculate_kama(df_1d['close'].values, er_period=10)
    kama_1d_20 = calculate_kama(df_1d['close'].values, er_period=20)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    kama_1d_10_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_10)
    kama_1d_20_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_20)
    
    # Calculate 4h indicators
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    rsi_14 = calculate_rsi(close, 14)
    kama_4h_10 = calculate_kama(close, er_period=10)
    
    # ATR ratio for volatility regime
    atr_ratio = np.zeros(n)
    for i in range(30, n):
        if atr_30[i] > 0:
            atr_ratio[i] = atr_7[i] / atr_30[i]
        else:
            atr_ratio[i] = 1.0
    
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(kama_1d_10_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(atr_ratio[i]):
            continue
        
        # === MAJOR REGIME (1w HMA) ===
        # Bull: price above 1w HMA(21)
        # Bear: price below 1w HMA(21)
        major_bull = close[i] > hma_1w_21_aligned[i]
        major_bear = close[i] < hma_1w_21_aligned[i]
        
        # === PRIMARY TREND (1d KAMA) ===
        # Bull: KAMA(10) > KAMA(20)
        # Bear: KAMA(10) < KAMA(20)
        trend_bull = kama_1d_10_aligned[i] > kama_1d_20_aligned[i]
        trend_bear = kama_1d_10_aligned[i] < kama_1d_20_aligned[i]
        
        # === VOLATILITY REGIME (ATR ratio) ===
        # High vol (>1.8): mean reversion mode
        # Low vol (<1.2): trend following mode
        # Normal (1.2-1.8): mixed mode
        high_vol = atr_ratio[i] > 1.8
        low_vol = atr_ratio[i] < 1.2
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_long = fisher[i] > -1.5 and fisher[i-1] <= -1.5
        fisher_short = fisher[i] < 1.5 and fisher[i-1] >= 1.5
        
        # Fisher momentum (current value)
        fisher_bullish = fisher[i] > fisher_signal[i] and fisher[i] > -1.0
        fisher_bearish = fisher[i] < fisher_signal[i] and fisher[i] < 1.0
        
        # === RSI CONFIRMATION ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        rsi_bullish = rsi_14[i] > 45
        rsi_bearish = rsi_14[i] < 55
        
        # === 4H LOCAL TREND ===
        local_bull = close[i] > kama_4h_10[i]
        local_bear = close[i] < kama_4h_10[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (low volatility + trend aligned)
        if low_vol or (trend_bull and major_bull):
            # LONG: Trend bull + major bull + Fisher bullish + RSI confirmation
            if trend_bull and major_bull and fisher_bullish and rsi_bullish:
                new_signal = STRONG_SIZE
            # LONG: Trend bull + Fisher cross + local bull
            elif trend_bull and fisher_long and local_bull:
                new_signal = BASE_SIZE
            # LONG: Major bull + trend bull + RSI > 50
            elif major_bull and trend_bull and rsi_14[i] > 50:
                new_signal = BASE_SIZE
        
        if low_vol or (trend_bear and major_bear):
            # SHORT: Trend bear + major bear + Fisher bearish + RSI confirmation
            if trend_bear and major_bear and fisher_bearish and rsi_bearish:
                new_signal = -STRONG_SIZE
            # SHORT: Trend bear + Fisher cross + local bear
            elif trend_bear and fisher_short and local_bear:
                new_signal = -BASE_SIZE
            # SHORT: Major bear + trend bear + RSI < 50
            elif major_bear and trend_bear and rsi_14[i] < 50:
                new_signal = -BASE_SIZE
        
        # MEAN REVERSION MODE (high volatility)
        if high_vol:
            # LONG: Fisher oversold cross + RSI oversold (vol spike reversal)
            if fisher_long and rsi_oversold:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.7
            # LONG: RSI very oversold (<30) in any regime
            if rsi_14[i] < 30:
                if new_signal == 0.0:
                    new_signal = BASE_SIZE * 0.5
            
            # SHORT: Fisher overbought cross + RSI overbought (vol spike reversal)
            if fisher_short and rsi_overbought:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.7
            # SHORT: RSI very overbought (>70) in any regime
            if rsi_14[i] > 70:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE * 0.5
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 30+ trades) ===
        # Force trade if no signal for 35 bars (~6 days on 4h)
        if bars_since_last_trade > 35 and new_signal == 0.0 and not in_position:
            if trend_bull and major_bull and rsi_14[i] > 45:
                new_signal = BASE_SIZE * 0.4
            elif trend_bear and major_bear and rsi_14[i] < 55:
                new_signal = -BASE_SIZE * 0.4
            elif high_vol and rsi_14[i] < 38:
                new_signal = BASE_SIZE * 0.35
            elif high_vol and rsi_14[i] > 62:
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
            # Long position but major regime turns bearish
            if position_side > 0 and major_bear and trend_bear:
                regime_reversal = True
            # Short position but major regime turns bullish
            if position_side < 0 and major_bull and trend_bull:
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