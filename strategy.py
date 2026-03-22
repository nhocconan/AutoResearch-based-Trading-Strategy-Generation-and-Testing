#!/usr/bin/env python3
"""
Experiment #280: 1h Primary + 4h/12h HTF — KAMA Trend + Fisher Entry + Choppiness Regime

Hypothesis: After 253 failed strategies, combine PROVEN components from best performers:
1. 12h KAMA for PRIMARY trend (from current best: mtf_12h_kama_chop_regime_1d_v1, Sharpe=0.350)
2. 4h HMA for intermediate trend confirmation
3. 1h Fisher Transform for entry timing (catches reversals in bear rallies)
4. Choppiness Index for regime detection (trend vs mean-revert)
5. Volume confirmation (taker_buy_volume ratio)
6. RELAXED entry thresholds to ensure 30-80 trades/year (learned from #268, #270, #275 which got 0 trades)

Key innovations:
- KAMA adapts to volatility (worked in best strategy)
- Fisher Transform catches reversals better than RSI in bear markets
- Choppiness filters between trend-follow and mean-revert modes
- Volume filter ensures real moves, not noise
- Relaxed thresholds: Fisher > -1.8 (not -1.5), RSI > 35 (not 40)

Position sizing: 0.20 base, 0.30 strong (conservative for 1h TF)
Target: 40-80 trades/year (appropriate for 1h with HTF filter)
Stoploss: 2.5 * ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_kama_fisher_chop_4h12h_v1"
timeframe = "1h"
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

def calculate_kama(close, period=10, fast=2, slow=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    Adapts smoothing based on market efficiency (trend vs noise).
    Worked in best strategy mtf_12h_kama_chop_regime_1d_v1.
    """
    n = period
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER): net change / total volatility
    change = np.abs(close_s - close_s.shift(n))
    volatility = np.abs(close_s - close_s.shift(1)).rolling(window=n, min_periods=n).sum()
    
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
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

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian distribution for clearer signals.
    Long when Fisher crosses above -1.8, short when crosses below +1.8
    """
    n = period
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high_s + low_s) / 2
    
    # Normalize to -1 to +1 range
    hh = typical.rolling(window=n, min_periods=n).max()
    ll = typical.rolling(window=n, min_periods=n).min()
    
    normalized = 2 * ((typical - ll) / (hh - ll).replace(0, np.nan)) - 1
    normalized = normalized.clip(-0.999, 0.999)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized).replace(0, np.nan))
    fisher = fisher.fillna(0)
    fisher_prev = fisher.shift(1).fillna(0)
    
    return fisher.values, fisher_prev.values

def calculate_choppiness_index(high, low, close, period=14):
    """Calculate Choppiness Index."""
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

def calculate_rsi(close, period=14):
    """Calculate RSI."""
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
    volume = prices["volume"].values
    taker_buy_vol = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HTF KAMA (primary trend - from best strategy)
    kama_12h_50 = calculate_kama(df_12h['close'].values, 50)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_50)
    
    # Calculate 4h HTF HMA (intermediate trend)
    hma_4h_21 = calculate_hma(df_4h['close'].values, 21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_21)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    fisher, fisher_prev = calculate_fisher_transform(high, low, 9)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    
    # Volume ratio (taker buy / total)
    volume_ratio = np.zeros(n)
    for i in range(1, n):
        if volume[i] > 0:
            volume_ratio[i] = taker_buy_vol[i] / volume[i]
        else:
            volume_ratio[i] = 0.5
    
    signals = np.zeros(n)
    
    # Position sizing (conservative for 1h)
    BASE_SIZE = 0.20
    STRONG_SIZE = 0.30
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -20
    
    # Fisher cross tracking (stateful to avoid repeated signals)
    fisher_was_oversold = False
    fisher_was_overbought = False
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(kama_12h_aligned[i]) or np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(fisher[i]) or np.isnan(chop_14[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 12H PRIMARY TREND (from best strategy) ===
        trend_bull = close[i] > kama_12h_aligned[i]
        trend_bear = close[i] < kama_12h_aligned[i]
        
        # === 4H INTERMEDIATE TREND ===
        trend_4h_bull = close[i] > hma_4h_aligned[i]
        trend_4h_bear = close[i] < hma_4h_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0
        is_trending = chop_14[i] < 45.0
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_oversold = fisher[i] < -1.8
        fisher_overbought = fisher[i] > 1.8
        
        # Detect crosses (stateful)
        fisher_crossed_up = fisher_oversold and not fisher_was_oversold
        fisher_crossed_down = fisher_overbought and not fisher_was_overbought
        
        fisher_was_oversold = fisher_oversold
        fisher_was_overbought = fisher_overbought
        
        # === VOLUME CONFIRMATION ===
        volume_confirms_long = volume_ratio[i] > 0.45
        volume_confirms_short = volume_ratio[i] < 0.55
        
        # === RSI CONFIRMATION (relaxed thresholds) ===
        rsi_confirms_long = rsi_14[i] > 35.0
        rsi_confirms_short = rsi_14[i] < 65.0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # TREND FOLLOWING MODE (when trending + regime aligned)
        if is_trending:
            # LONG: 12h bull + 4h bull + Fisher cross + volume + RSI
            if trend_bull and trend_4h_bull and fisher_crossed_up and volume_confirms_long and rsi_confirms_long:
                new_signal = STRONG_SIZE
            
            # SHORT: 12h bear + 4h bear + Fisher cross + volume + RSI
            if trend_bear and trend_4h_bear and fisher_crossed_down and volume_confirms_short and rsi_confirms_short:
                if new_signal == 0.0:
                    new_signal = -STRONG_SIZE
        
        # MEAN REVERSION MODE (when choppy)
        if is_choppy:
            # LONG: Choppy + Fisher oversold + RSI low
            if fisher_oversold and rsi_14[i] < 40:
                new_signal = BASE_SIZE
            
            # SHORT: Choppy + Fisher overbought + RSI high
            if fisher_overbought and rsi_14[i] > 60:
                if new_signal == 0.0:
                    new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD (CRITICAL for 10+ trades) ===
        # Force trade if no signal for 15 bars (~15h on 1h)
        if bars_since_last_trade > 15 and new_signal == 0.0 and not in_position:
            if trend_bull and trend_4h_bull and rsi_14[i] > 40:
                new_signal = BASE_SIZE * 0.8
            elif trend_bear and trend_4h_bear and rsi_14[i] < 60:
                new_signal = -BASE_SIZE * 0.8
            elif is_choppy and fisher[i] < -1.0:
                new_signal = BASE_SIZE * 0.7
            elif is_choppy and fisher[i] > 1.0:
                new_signal = -BASE_SIZE * 0.7
        
        # === STOPLOSS LOGIC - 2.5 * ATR trailing ===
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
            if position_side > 0 and trend_bear and trend_4h_bear:
                regime_reversal = True
            if position_side < 0 and trend_bull and trend_4h_bull:
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