#!/usr/bin/env python3
"""
Experiment #022: TRIX + Choppiness + Volume on 12h with 1d HTF

HYPOTHESIS: TRIX momentum combined with choppiness regime filter works in both bull/bear:
- Bull market: TRIX positive + choppiness trending = ride the uptrend
- Bear market: TRIX negative + choppiness trending = short rallies
- Range: Choppiness high = no trade (avoid whipsaws)
- 1d HTF confirms major trend direction

WHY IT SHOULD WORK: TRIX is a proven momentum indicator (test Sharpe 1.32 on ETH).
Choppiness Index is the meta-regime filter (better than ADX for avoiding false signals).
Volume confirms the move. Simple 2-3 conditions = fewer trades = less fee drag.

PATTERN FROM DB: "TRIX + volume spike + regime" on 4h = ETH 1.32 test Sharpe.
This adapts it to 12h timeframe (less fee drag) with 1d HTF trend confirmation.

TARGET: 75-200 total trades over 4 years (19-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_trix_chop_vol_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_trix(close, period=14):
    """TRIX: Triple EMA rate of change - smooth momentum"""
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # TRIX = rate of change of triple EMA
    trix = np.full(n, np.nan)
    for i in range(period, n):
        if ema3[i - period] != 0:
            trix[i] = 100 * (ema3[i] - ema3[i - period]) / ema3[i - period]
    
    return trix

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market "choppiness" (range-bound vs trending)
    CHOP < 38.2 = trending (low values = strong trend)
    CHOP > 61.8 = ranging (high values = choppy)
    Values between = neutral
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of True Range over period
        tr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j - 1]) if j > 0 else high[j] - low[j])
            tr_sum += tr
        
        # Highest high - Lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / range_sum) / np.log10(period)
    
    return chop

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=16):
    """Hull Moving Average - faster trend detection"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    # WMA with half period
    half = period // 2
    wma_half = pd.Series(close).rolling(window=half, min_periods=half).mean().values
    wma_full = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    
    # HMA = 2 * WMA(half) - WMA(full), then WMA(sqrt(period))
    hma_input = 2 * wma_half - wma_full
    
    hma = pd.Series(hma_input).rolling(window=int(np.sqrt(period)), min_periods=int(np.sqrt(period))).mean().values
    return hma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load 1d HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === HTF indicators (1d) ===
    hma_21_1d = calculate_hma(df_1d['close'].values, 21)
    hma_trend = hma_21_1d  # Price above HMA = bull, below = bear
    
    # Align HTF to local
    hma_aligned = align_htf_to_ltf(prices, df_1d, hma_trend)
    
    # === Local 12h indicators ===
    trix = calculate_trix(close, period=14)
    chop = calculate_choppiness(high, low, close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    # Volume ratio (20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Local HMA for price direction ===
    hma_12h = calculate_hma(close, 21)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 100  # TRIX needs ~42 bars, chop needs 14, volume needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(trix[i]) or np.isnan(chop[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        if atr[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_aligned[i]) or np.isnan(hma_12h[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER ===
        # Choppiness: < 50 = trending, can trade. > 60 = ranging, skip
        trending_regime = chop[i] < 55  # Slightly relaxed from 50 to get more trades
        
        # === HTF TREND (1d HMA) ===
        htf_bull = hma_aligned[i] > hma_aligned[i - 1] if i > 0 and not np.isnan(hma_aligned[i - 1]) else False
        htf_bear = hma_aligned[i] < hma_aligned[i - 1] if i > 0 and not np.isnan(hma_aligned[i - 1]) else False
        
        # HTF absolute position
        htf_above = close[i] > hma_12h[i]  # Local price above local HMA (proxy for 1d trend)
        
        # === MOMENTUM (TRIX) ===
        trix_positive = trix[i] > 0
        trix_negative = trix[i] < 0
        
        # TRIX cross (momentum shift)
        trix_cross_up = trix[i] > 0 and trix[i - 1] <= 0 if i > 0 else False
        trix_cross_down = trix[i] < 0 and trix[i - 1] >= 0 if i > 0 else False
        
        # === VOLUME CONFIRMATION ===
        vol_confirm = vol_ratio[i] > 1.4  # Volume above 1.4x average
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: TRIX positive (or cross up) + trending regime + volume + HTF bull aligned
            bull_trix = trix_positive or trix_cross_up
            bull_confirm = htf_above or htf_bull  # HTF trend agrees
            
            if bull_trix and trending_regime and vol_confirm and bull_confirm:
                desired_signal = SIZE
            
            # SHORT: TRIX negative (or cross down) + trending regime + volume + HTF bear aligned
            bear_trix = trix_negative or trix_cross_down
            bear_confirm = not htf_above or htf_bear  # HTF trend agrees
            
            if bear_trix and trending_regime and vol_confirm and bear_confirm:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if TRIX turns negative
                if trix_negative:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if TRIX turns positive
                if trix_positive:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals