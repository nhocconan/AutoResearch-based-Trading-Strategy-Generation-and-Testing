#!/usr/bin/env python3
"""
Experiment #1214: 4h Primary + 12h/1d HTF — Fisher Transform + Choppiness Regime + HMA Trend

Hypothesis: 4h timeframe is proven (current best Sharpe=0.612). Adding Ehlers Fisher Transform
for reversal detection in bear markets, combined with Choppiness Index regime switching and
HTF HMA trend filter. Fisher Transform normalizes price to Gaussian distribution, making
extreme reversals more detectable than RSI alone.

Key design:
- 4h primary for optimal trade frequency (20-50/year target)
- 12h HMA for intermediate trend direction
- 1d HMA for macro bias (looser filter than 1w)
- Fisher Transform(9) for entry timing (crosses at -1.5/+1.5 levels)
- Choppiness Index(14) for regime: >61.8 chop (mean revert), <38.2 trend (breakout)
- RSI(14) as secondary confirmation (looser thresholds: 30/70)
- ATR(14) 2.5x trailing stop for risk management
- Position size: 0.28 discrete (conservative for 4h)
- Asymmetric bias: favor shorts when 1d HMA bearish (2025+ is bear market)

Entry logic (LOOSENED for more trades):
- Trend regime: Fisher cross + HMA alignment + RSI confirmation
- Chop regime: Fisher extreme + RSI extreme (either triggers entry)
- Transition: Use Fisher direction + HMA slope

Target: Sharpe > 0.612, trades >= 30 on train, >= 3 on test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_regime_12h1d_hma_rsi_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - normalizes price to Gaussian distribution.
    Catches reversals better than RSI in bear/range markets.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    if n < period + 1:
        return fisher, fisher_prev
    
    for i in range(period, n):
        # Calculate price range
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        price_range = highest - lowest
        
        if price_range < 1e-10:
            fisher[i] = fisher[i - 1] if i > period else 0.0
            fisher_prev[i] = fisher[i - 1] if i > period else 0.0
            continue
        
        # Normalize price to 0-1 range
        x = (close[i] - lowest) / price_range
        
        # Clamp to avoid log(0)
        x = max(0.001, min(0.999, x))
        
        # Fisher transform
        fisher[i] = 0.5 * np.log((1 + x) / (1 - x))
        
        # Smooth with previous value
        if i > period and not np.isnan(fisher[i - 1]):
            fisher[i] = 0.67 * fisher[i] + 0.33 * fisher_prev[i - 1]
        
        fisher_prev[i] = fisher[i]
    
    return fisher, fisher_prev

def calculate_hma(close, period=21):
    """Hull Moving Average — reduces lag while maintaining smoothness."""
    n = len(close)
    hma = np.full(n, np.nan)
    
    if n < period:
        return hma
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_window = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_window.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_window) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1)
                    hma[i] = np.sum(np.array(diff_window) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index — momentum oscillator."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    if n < period + 1:
        return rsi
    
    delta = np.diff(close)
    gain = np.zeros(n)
    loss = np.zeros(n)
    
    gain[1:] = np.where(delta > 0, delta, 0)
    loss[1:] = np.where(delta < 0, -delta, 0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    mask = loss_smooth > 1e-10
    rs = np.zeros(n)
    rs[mask] = gain_smooth[mask] / loss_smooth[mask]
    
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    
    return rsi

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index — measures market choppiness vs trending.
    CHOP > 61.8 = choppy/range (mean revert)
    CHOP < 38.2 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and atr_sum > 1e-10:
            chop[i] = 100.0 * np.log10(atr_sum / price_range) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range for volatility measurement and stoploss."""
    n = len(close)
    atr = np.full(n, np.nan)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 12h HMA for intermediate trend
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 1d HMA for macro trend filter
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or np.isnan(chop[i]) or np.isnan(rsi[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        if atr[i] <= 1e-10:
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === INTERMEDIATE TREND (12h HMA) ===
        inter_bull = close[i] > hma_12h_aligned[i]
        inter_bear = close[i] < hma_12h_aligned[i]
        
        # === 12h HMA SLOPE ===
        hma_12h_slope_up = False
        hma_12h_slope_down = False
        if i >= 5 and not np.isnan(hma_12h_aligned[i-5]):
            hma_12h_slope_up = hma_12h_aligned[i] > hma_12h_aligned[i-5]
            hma_12h_slope_down = hma_12h_aligned[i] < hma_12h_aligned[i-5]
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop[i] > 61.8
        is_trending = chop[i] < 38.2
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long_cross = fisher_prev[i] < -1.5 and fisher[i] >= -1.5
        fisher_short_cross = fisher_prev[i] > 1.5 and fisher[i] <= 1.5
        fisher_extreme_low = fisher[i] < -1.8
        fisher_extreme_high = fisher[i] > 1.8
        
        # === RSI EXTREMES (looser for more trades) ===
        rsi_oversold = rsi[i] < 35.0
        rsi_overbought = rsi[i] > 65.0
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # === TRENDING REGIME: Follow HMA direction + Fisher confirmation ===
        if is_trending:
            # Long: 12h HMA up + Fisher long cross or extreme low + RSI not overbought
            if hma_12h_slope_up and (fisher_long_cross or fisher_extreme_low) and not rsi_overbought:
                desired_signal = BASE_SIZE
            # Short: 12h HMA down + Fisher short cross or extreme high + RSI not oversold
            elif hma_12h_slope_down and (fisher_short_cross or fisher_extreme_high) and not rsi_oversold:
                desired_signal = -BASE_SIZE
        
        # === CHOPPY REGIME: Mean Reversion (Fisher extremes + RSI) ===
        elif is_choppy:
            # Long: Fisher extreme low OR (Fisher cross + RSI oversold)
            if fisher_extreme_low or (fisher_long_cross and rsi_oversold):
                desired_signal = BASE_SIZE
            # Short: Fisher extreme high OR (Fisher cross + RSI overbought)
            elif fisher_extreme_high or (fisher_short_cross and rsi_overbought):
                desired_signal = -BASE_SIZE
        
        # === TRANSITION ZONE (38.2 <= CHOP <= 61.8): Use 12h HMA + Fisher ===
        else:
            # Long: 12h HMA bullish + Fisher turning up
            if inter_bull and fisher[i] > fisher_prev[i] and fisher[i] < 0:
                desired_signal = BASE_SIZE
            # Short: 12h HMA bearish + Fisher turning down
            elif inter_bear and fisher[i] < fisher_prev[i] and fisher[i] > 0:
                desired_signal = -BASE_SIZE
        
        # === ASYMMETRIC BIAS (bear market preference for shorts) ===
        # In bear macro, reduce long size and increase short conviction
        if macro_bear and desired_signal > 0:
            desired_signal = BASE_SIZE * 0.7  # Reduce long size in bear market
        elif macro_bull and desired_signal < 0:
            desired_signal = -BASE_SIZE * 0.7  # Reduce short size in bull market
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.15:
            desired_signal = BASE_SIZE
        elif desired_signal < -0.15:
            desired_signal = -BASE_SIZE
        else:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals