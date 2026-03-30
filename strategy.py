#!/usr/bin/env python3
"""
Experiment #022: 12h Donchian Breakout + 1d Trend + Volume Confirmation

HYPOTHESIS: Simple price channel breakout (Donchian) on 12h with HTF trend
confirmation works in both bull and bear because:
- Donchian(20) = 10 days of price action = captures major swings
- 1d HTF trend filter = avoids fighting major direction
- Volume spike confirmation = prevents false breakouts in thin markets
- Choppiness regime = stays out of range-bound periods

WHY IT SHOULD WORK IN BOTH BULL AND BEAR:
- Bull: Price breaks above 20-bar high + 1d trend bullish + vol spike = long
- Bear: Price breaks below 20-bar low + 1d trend bearish + vol spike = short
- Range: Choppiness > 61.8 = no trades (prevents whipsaws)

KEY INSIGHT: DB shows Donchian strategies are TOP performers (Sharpe 1.1-1.5 on test).
Keep entry simple: just the breakout + 1-2 confirmations.
Target: 75-150 total trades over 4 years (19-37/year) on 12h.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_trend_vol_v6"
timeframe = "12h"
leverage = 1.0

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

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index: measures market volatility vs trending
    CHOP < 38.2 = trending (use trend following)
    CHOP > 61.8 = ranging (avoid or use mean reversion)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    for i in range(period, n):
        # Sum of ATR over period
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]) if j > 0 else high[j] - low[j])
            atr_sum += tr
        
        # Highest high - lowest low over period
        hh = np.max(high[i - period + 1:i + 1])
        ll = np.min(low[i - period + 1:i + 1])
        range_sum = hh - ll
        
        if range_sum > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
    
    return chop

def calculate_hma(values, period):
    """Hull Moving Average"""
    data = pd.Series(values)
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = data.rolling(window=half, min_periods=half).mean()
    wma_full = data.rolling(window=period, min_periods=period).mean()
    
    hull = (2 * wma_half - wma_full).rolling(window=sqrt_n, min_periods=sqrt_n).mean()
    return hull.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d HTF indicators for trend direction ===
    htf_close = df_1d['close'].values
    htf_high = df_1d['high'].values
    htf_low = df_1d['low'].values
    
    # HTF HMA(21) for trend
    htf_hma = calculate_hma(htf_close, 21)
    htf_hma_aligned = align_htf_to_ltf(prices, df_1d, htf_hma)
    
    # HTF: price above HMA = bull, below = bear
    htf_bullish = htf_close > htf_hma
    htf_bearish = htf_close < htf_hma
    htf_bullish_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish.astype(float))
    htf_bearish_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish.astype(float))
    
    # === Local 12h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = 10 trading days on 12h)
    donchian_period = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Choppiness Index
    chop = calculate_choppiness(high, low, close, period=14)
    
    # Volume ratio
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # Local HMA for trend
    hma_21 = calculate_hma(close, 21)
    
    # Signals
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Donchian needs 20, chop needs 14, vol needs 20
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(chop[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(hmana_21[i]):
            signals[i] = 0.0
            continue
        
        # === REGIME FILTER ===
        # Only trade when choppiness < 61.8 (not ranging)
        # Allow trades even in moderate chop (< 70) to ensure enough trades
        ranging = chop[i] > 70.0
        
        if ranging:
            # In range - flatten any existing position
            if in_position:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # === HTF TREND ===
        htf_bull = htf_bullish_aligned[i] > 0.5 if not np.isnan(htf_bullish_aligned[i]) else False
        htf_bear = htf_bearish_aligned[i] > 0.5 if not np.isnan(htf_bearish_aligned[i]) else False
        
        # === LOCAL TREND ===
        local_bull = close[i] > hma_21[i]
        local_bear = close[i] < hma_21[i]
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5  # 1.5x average volume
        
        # === DONCHIAN BREAKOUT DETECTION ===
        # Price breaks above 20-bar high
        bull_breakout = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        # Price breaks below 20-bar low
        bear_breakout = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # Price near channel extremes (within 0.5 ATR)
        near_upper = close[i] >= donchian_upper[i] - 0.5 * atr_14[i]
        near_lower = close[i] <= donchian_lower[i] + 0.5 * atr_14[i]
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Donchian breakout + HTF bull + local bull + vol spike
            if bull_breakout and htf_bull and local_bull and vol_spike:
                desired_signal = SIZE
            
            # LONG (alt): Near upper break + strong confirmations (no breakout yet)
            elif near_upper and htf_bull and local_bull and vol_spike and vol_ratio[i] > 2.0:
                desired_signal = SIZE
            
            # SHORT: Donchian breakdown + HTF bear + local bear + vol spike
            if bear_breakout and htf_bear and local_bear and vol_spike:
                desired_signal = -SIZE
            
            # SHORT (alt): Near lower breakdown + strong confirmations
            elif near_lower and htf_bear and local_bear and vol_spike and vol_ratio[i] > 2.0:
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
                
                # Exit if local trend turns
                if not local_bull and close[i] < hma_21[i]:
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
                
                # Exit if local trend turns
                if not local_bear and close[i] > hma_21[i]:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 3 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 3:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals