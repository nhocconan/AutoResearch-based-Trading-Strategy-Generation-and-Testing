#!/usr/bin/env python3
"""
Experiment #504: 4h Primary + 1d HTF — Funding Rate Mean Reversion + Vol Regime + HMA Trend

Hypothesis: After 499 failed experiments (mostly CRSI/Choppiness/HMA combos), try the 
MOST PROVEN edge from research notes: FUNDING RATE MEAN REVERSION.

Research states: "FUNDING RATE MEAN REVERSION: Z-score of funding(30d) < -2 → long, 
> +2 → short. Reported Sharpe 0.8-1.5 through 2022 crash. BEST EDGE for BTC/ETH."

Since funding data may not be available in all environments, I'll use a PRICE-BASED 
PROXY that captures the same mean-reversion dynamic:

1. PRICE Z-SCORE (20-period): When price deviates >2.5 std from SMA, expect reversion
   This mimics funding rate extremes (over-leveraged longs/shorts)
   
2. VOL REGIME FILTER: ATR(7)/ATR(30) ratio determines if we're in panic (mean revert)
   or calm (trend follow) mode
   
3. 1D HMA TREND: Only take mean reversion trades WITH the major trend
   Bull regime: long only at extremes. Bear regime: short only at extremes
   
4. FISHER TRANSFORM: Precise entry timing on reversals (proven for bear markets)

Why this might beat current best (Sharpe=0.435):
- Funding rate mean reversion is THE MOST PROVEN edge for BTC/ETH perpetuals
- Price z-score proxy captures the same overcrowding dynamics
- Vol regime filter prevents trading mean reversion in strong trends
- 4h TF = 20-50 trades/year target (lower fee drag than 1h/30m)
- Simpler logic = more trades (critical: need >=30/symbol on train)

Position sizing: 0.25-0.30 (discrete levels, max 0.40)
Stoploss: 2.5 * ATR trailing (signal → 0 when hit)
Target: 25-50 trades/year on 4h, >=30 trades/symbol on train, >=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_funding_proxy_volregime_hma_1d_v1"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    n = period
    half = n // 2
    sqrt_n = int(np.sqrt(n))
    
    close_s = pd.Series(close)
    
    def wma(series, span):
        weights = np.arange(1, span + 1)
        return series.rolling(window=span, min_periods=span).apply(
            lambda x: np.dot(x, weights) / weights.sum(), raw=True
        )
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    hma_raw = 2.0 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_n)
    
    return hma.values

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    typical = (high_s + low_s) / 2.0
    highest = typical.rolling(window=period, min_periods=period).max()
    lowest = typical.rolling(window=period, min_periods=period).min()
    
    range_hl = highest - lowest
    range_hl = range_hl.replace(0, 1e-10)
    
    normalized = (typical - lowest) / range_hl
    normalized = np.clip(normalized * 2.0 - 1.0, -0.999, 0.999)
    
    fisher = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
    
    return fisher.values

def calculate_price_zscore(close, period=20):
    """
    Calculate price z-score (deviation from SMA in std units).
    Proxy for funding rate extremes - when price is far from mean, 
    overcrowded positions expect mean reversion.
    """
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    zscore = (close_s - sma) / (std + 1e-10)
    
    return zscore.values

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    std = close_s.rolling(window=period, min_periods=period).std()
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper.values, lower.values, sma.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators (major trend direction)
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_50 = calculate_hma(df_1d['close'].values, period=50)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    
    # Vol regime: ATR(7) / ATR(30)
    vol_ratio = atr_7 / (atr_30 + 1e-10)
    
    # Price z-score (funding rate proxy)
    price_zscore = calculate_price_zscore(close, period=20)
    
    # Fisher Transform for entry timing
    fisher = calculate_fisher_transform(high, low, period=9)
    
    # Bollinger Bands for extreme detection
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    
    # RSI for confirmation
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    LONG_SIZE = 0.30
    SHORT_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Track Fisher for crossover detection
    prev_fisher = np.zeros(n)
    prev_fisher[1:] = fisher[:-1]
    
    for i in range(300, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1d_50_aligned[i]):
            continue
        if np.isnan(vol_ratio[i]) or np.isnan(price_zscore[i]):
            continue
        if np.isnan(fisher[i]) or np.isnan(bb_lower[i]):
            continue
        
        # === 1D MAJOR TREND (primary direction filter) ===
        bull_regime = close[i] > hma_1d_21_aligned[i]
        bear_regime = close[i] < hma_1d_21_aligned[i]
        
        # 1d HMA slope for trend strength
        hma_slope_bull = hma_1d_21_aligned[i] > hma_1d_50_aligned[i]
        hma_slope_bear = hma_1d_21_aligned[i] < hma_1d_50_aligned[i]
        
        # === VOL REGIME DETECTION ===
        # High vol ratio = panic/extreme (mean revert)
        # Low vol ratio = calm (trend follow)
        high_vol_regime = vol_ratio[i] > 1.8
        low_vol_regime = vol_ratio[i] < 1.2
        
        # === PRICE Z-SCORE (FUNDING RATE PROXY) ===
        # Z-score > 2.5 = overcrowded longs (expect short mean reversion)
        # Z-score < -2.5 = overcrowded shorts (expect long mean reversion)
        zscore_extreme_high = price_zscore[i] > 2.0
        zscore_extreme_low = price_zscore[i] < -2.0
        zscore_moderate_high = price_zscore[i] > 1.5
        zscore_moderate_low = price_zscore[i] < -1.5
        
        # === FISHER TRANSFORM REVERSAL SIGNALS ===
        fisher_cross_up = (fisher[i] > -1.0) and (prev_fisher[i] <= -1.0)
        fisher_cross_down = (fisher[i] < 1.0) and (prev_fisher[i] >= 1.0)
        fisher_extreme_low = fisher[i] < -1.5
        fisher_extreme_high = fisher[i] > 1.5
        
        # === BOLLINGER BAND EXTREMES ===
        bb_extreme_low = close[i] < bb_lower[i]
        bb_extreme_high = close[i] > bb_upper[i]
        
        # === RSI EXTREMES ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_low = rsi_14[i] < 25.0
        rsi_extreme_high = rsi_14[i] > 75.0
        
        # === ENTRY LOGIC — FUNDING PROXY + VOL REGIME + TREND ===
        new_signal = 0.0
        
        # LONG ENTRIES (mean reversion in bull regime, or extreme oversold in any regime)
        # Condition 1: Bull regime + z-score extreme low + Fisher cross up (primary signal)
        if bull_regime and zscore_extreme_low and fisher_cross_up:
            new_signal = LONG_SIZE
        # Condition 2: Bull regime + BB extreme low + RSI oversold (confluence)
        elif bull_regime and bb_extreme_low and rsi_oversold:
            new_signal = LONG_SIZE
        # Condition 3: High vol regime + z-score extreme low + RSI extreme (panic bottom)
        elif high_vol_regime and zscore_extreme_low and rsi_extreme_low:
            new_signal = LONG_SIZE
        # Condition 4: Any regime + extreme z-score + Fisher extreme (strong mean revert)
        elif zscore_extreme_low and fisher_extreme_low:
            new_signal = LONG_SIZE * 0.8
        # Condition 5: Bull regime + moderate z-score low + Fisher cross (pullback entry)
        elif bull_regime and zscore_moderate_low and fisher_cross_up:
            new_signal = LONG_SIZE * 0.7
        
        # SHORT ENTRIES (mirror logic for bear market)
        if new_signal == 0.0:
            # Condition 1: Bear regime + z-score extreme high + Fisher cross down
            if bear_regime and zscore_extreme_high and fisher_cross_down:
                new_signal = -SHORT_SIZE
            # Condition 2: Bear regime + BB extreme high + RSI overbought
            elif bear_regime and bb_extreme_high and rsi_overbought:
                new_signal = -SHORT_SIZE
            # Condition 3: High vol regime + z-score extreme high + RSI extreme (panic top)
            elif high_vol_regime and zscore_extreme_high and rsi_extreme_high:
                new_signal = -SHORT_SIZE
            # Condition 4: Any regime + extreme z-score + Fisher extreme
            elif zscore_extreme_high and fisher_extreme_high:
                new_signal = -SHORT_SIZE * 0.8
            # Condition 5: Bear regime + moderate z-score high + Fisher cross
            elif bear_regime and zscore_moderate_high and fisher_cross_down:
                new_signal = -SHORT_SIZE * 0.7
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === TAKE PROFIT / EXIT CONDITIONS ===
        # Exit long on z-score mean reversion or Fisher extreme high
        if in_position and position_side > 0:
            # Take profit when z-score reverts to neutral
            if price_zscore[i] > 0.5:
                new_signal = 0.0
            # Exit on Fisher extreme or RSI overbought
            elif fisher_extreme_high or rsi_overbought:
                new_signal = 0.0
            # Exit if regime flips strongly bearish
            elif bear_regime and hma_slope_bear:
                new_signal = 0.0
        
        # Exit short on z-score mean reversion or Fisher extreme low
        if in_position and position_side < 0:
            # Take profit when z-score reverts to neutral
            if price_zscore[i] < -0.5:
                new_signal = 0.0
            # Exit on Fisher extreme or RSI oversold
            elif fisher_extreme_low or rsi_oversold:
                new_signal = 0.0
            # Exit if regime flips strongly bullish
            elif bull_regime and hma_slope_bull:
                new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Flip position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals