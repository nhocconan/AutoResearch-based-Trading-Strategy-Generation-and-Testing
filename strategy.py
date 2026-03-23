#!/usr/bin/env python3
"""
Experiment #1264: 4h Primary + 12h HTF — KAMA Trend + Fisher Reversal + Volume

Hypothesis: Recent failures (#1254-1263) show either ZERO TRADES or negative Sharpe.
The problem is over-filtering. This strategy uses:
1. 12h HMA for MACRO trend filter (only trade with higher TF direction)
2. 4h KAMA for adaptive local trend (responds to volatility changes)
3. Ehlers Fisher Transform for entry timing (proven reversal catcher)
4. Volume confirmation (avoid low-liquidity false signals)
5. LOOSE entry thresholds to ensure >=30 trades/symbol/train

Key changes from #1254:
- Replace CRSI with Fisher Transform (fewer parameters, more reliable)
- Replace HMA with KAMA (adapts to market regime automatically)
- Remove Choppiness Index (was creating binary regime whipsaw)
- Lower Fisher thresholds (-1.2/+1.2 vs -1.5/+1.5) for MORE trades
- Add simple volume filter (vol > 0.8 * avg) not strict (>1.5 * avg)
- Fixed ATR stoploss from entry (not trailing = less churn)

Target: Sharpe > 0.612, trades >= 60 train, >= 8 test
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_vol_12h_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """Kaufman Adaptive Moving Average - adapts to market efficiency
    KAMA responds quickly in trending markets, slowly in ranging markets
    """
    n = len(close)
    kama = np.full(n, np.nan)
    
    if n < er_period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        if change < 1e-10:
            er[i] = 0.0
        else:
            volatility = np.sum(np.abs(np.diff(close[i - er_period:i + 1])))
            if volatility > 1e-10:
                er[i] = change / volatility
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if not np.isnan(er[i]):
            sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
            kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher(close, period=9):
    """Ehlers Fisher Transform - normalizes price to -1 to +1
    Long when Fisher crosses above -1.5 from below
    Short when Fisher crosses below +1.5 from above
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher, fisher_signal
    
    for i in range(period, n):
        # Find highest high and lowest low over period
        hh = np.max(close[i - period + 1:i + 1])
        ll = np.min(close[i - period + 1:i + 1])
        
        if hh > ll:
            # Normalize price to -1 to +1
            normalized = 2.0 * (close[i] - ll) / (hh - ll) - 1.0
            normalized = np.clip(normalized, -0.999, 0.999)
            
            # Fisher transform
            fisher[i] = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
            
            if i > period:
                fisher_signal[i] = fisher[i - 1]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
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

def calculate_volume_sma(volume, period=20):
    """Simple moving average of volume"""
    n = len(volume)
    vol_sma = np.full(n, np.nan)
    
    if n < period:
        return vol_sma
    
    for i in range(period - 1, n):
        vol_sma[i] = np.mean(volume[i - period + 1:i + 1])
    
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for macro trend filter
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    kama = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher(close, period=9)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(kama[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            continue
        if np.isnan(vol_sma[i]) or vol_sma[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === MACRO TREND (12h HMA) ===
        macro_bull = close[i] > hma_12h_aligned[i]
        macro_bear = close[i] < hma_12h_aligned[i]
        
        # === LOCAL TREND (4h KAMA) ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === VOLUME FILTER (loose - just avoid dead periods) ===
        vol_ok = volume[i] > 0.7 * vol_sma[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        fisher_long = fisher_signal[i] < -1.2 and fisher[i] > -1.2
        fisher_short = fisher_signal[i] > 1.2 and fisher[i] < 1.2
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long: Macro bull + KAMA bull + Fisher reversal + Volume OK
        if macro_bull and kama_bull and fisher_long and vol_ok:
            desired_signal = BASE_SIZE
        
        # Short: Macro bear + KAMA bear + Fisher reversal + Volume OK
        elif macro_bear and kama_bear and fisher_short and vol_ok:
            desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Fixed 2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position:
            if position_side > 0 and close[i] < stop_price:
                stoploss_triggered = True
            elif position_side < 0 and close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal > 0.1:
            final_signal = BASE_SIZE
        elif desired_signal < -0.1:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                stop_price = entry_price - 2.5 * entry_atr if position_side > 0 else entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
        
        signals[i] = final_signal
    
    return signals