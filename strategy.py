#!/usr/bin/env python3
"""
Experiment #007: 6h Donchian Breakout + 1d SMA200 Trend + Volume

HYPOTHESIS: Simple 3-condition breakout on 6h timeframe:
1. Price breaks 20-bar Donchian high/low
2. Confirmed by 1d SMA200 trend direction
3. Volume spike > 1.5x 20-bar average

WHY IT WORKS IN BOTH BULL AND BEAR:
- Bull market: Price > SMA200 + breakout above 20-high = ride momentum higher
- Bear market: Price < SMA200 + breakdown below 20-low = short the continuation
- Range/bear: Price > SMA200 during breakdown = avoid bearish traps

SIMPLICITY = RELIABILITY. Fewer conditions = more consistent trades.
Target: 80-150 total trades over 4 years (20-37/year) on 6h.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_sma200_vol_v1"
timeframe = "6h"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d SMA(200) for trend direction ===
    sma_200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    price_1d = df_1d['close'].values
    
    # HTF: trend = 1 if bull, -1 if bear, 0 if neutral
    htf_bullish = (price_1d > sma_200_1d).astype(float)
    htf_bearish = (price_1d < sma_200_1d).astype(float)
    
    # Align HTF to 6h with shift(1) to avoid look-ahead
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bullish)
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bearish)
    
    # === Local 6h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian Channels (20 periods)
    donchian_period = 20
    rolling_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    rolling_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
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
    
    warmup = 250  # SMA200 on 1d needs 200*4 = 800 6h bars, but aligned so just 200 1d bars
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(rolling_high[i]) or np.isnan(rolling_low[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(htf_bull_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Breakout high (close above 20-bar high)
        breakout_high = close[i] > rolling_high[i] and close[i-1] <= rolling_high[i-1]
        # Breakdown low (close below 20-bar low)
        breakdown_low = close[i] < rolling_low[i] and close[i-1] >= rolling_low[i-1]
        
        # === HTF TREND ===
        htf_bull = htf_bull_aligned[i] > 0.5
        htf_bear = htf_bear_aligned[i] > 0.5
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC (3 conditions) ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Breakout high + volume spike + HTF bull (or neutral)
            # Entry: close breaks above 20-bar high
            # Confirmation: volume spike + HTF trend aligns
            if breakout_high and vol_spike:
                if htf_bull:  # Must be in bull trend
                    desired_signal = SIZE
            
            # SHORT: Breakdown low + volume spike + HTF bear (or neutral)  
            # Entry: close breaks below 20-bar low
            # Confirmation: volume spike + HTF trend aligns
            if breakdown_low and vol_spike:
                if htf_bear:  # Must be in bear trend
                    desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing stop) ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Trailing stop: exit if price falls 2.5 ATR from peak
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Also exit if HTF turns bearish
                if htf_bear:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Trailing stop: exit if price rises 2.5 ATR from trough
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Also exit if HTF turns bullish
                if htf_bull:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 6 bars (1.5 days) to avoid fee churn ===
        if in_position and (i - entry_bar) < 6:
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