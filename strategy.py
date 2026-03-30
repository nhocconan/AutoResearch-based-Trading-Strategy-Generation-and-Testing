#!/usr/bin/env python3
"""
Experiment #027: Williams %R Extreme + Donchian Break + Volume (4h)

HYPOTHESIS: Williams %R extreme zone crossovers combined with Donchian
breakout structure and volume confirmation create high-probability entries.

WHY IT SHOULD WORK IN BULL AND BEAR:
- Bull: %R crosses above -20 (oversold recovery) + price breaks Donchian high = strong momentum long
- Bear: %R crosses below -80 (overbought breakdown) + price breaks Donchian low = strong momentum short
- Range: %R reversals catch mean-reversion bounces in both directions
- Both bull and bear have symmetric entry logic via %R extremes

Williams %R is a proven momentum oscillator that works in both trending and ranging markets.
The -20 (overbought) and -80 (oversold) thresholds are classic extreme zones.
Combined with Donchian breakout for structure and volume confirmation.

KEY INSIGHT: This is a NOVEL combination not extensively tested in the DB.
Williams %R + Donchian breakout is simpler than Ichimoku/Alligator = more reliable trades.

TARGET: 100-200 total trades over 4 years (25-50/year)
SIZE: 0.28
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_willr_donchian_vol_v3"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - returns upper, middle, lower"""
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(n):
        start = max(0, i - period + 1)
        upper[i] = np.max(high[start:i+1])
        lower[i] = np.min(low[start:i+1])
    
    return upper, lower

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator
    -20 to 0 = overbought
    -80 to -100 = oversold
    """
    n = len(close)
    willr = np.zeros(n)
    
    for i in range(n):
        if i < period - 1:
            willr[i] = -50
            continue
        
        start = max(0, i - period + 1)
        period_high = np.max(high[start:i+1])
        period_low = np.min(low[start:i+1])
        
        if period_high != period_low:
            willr[i] = -100 * (period_high - close[i]) / (period_high - period_low)
        else:
            willr[i] = -50
    
    return willr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # HTF: SMA(50) on 1d for trend direction
    htf_close = df_1d['close'].values
    htf_sma50 = pd.Series(htf_close).ewm(span=50, min_periods=30, adjust=False).mean().values
    htf_bull = htf_close > htf_sma50
    htf_bear = htf_close < htf_sma50
    
    # Align HTF trend to 4h
    htf_bull_aligned = align_htf_to_ltf(prices, df_1d, htf_bull.astype(float))
    htf_bear_aligned = align_htf_to_ltf(prices, df_1d, htf_bear.astype(float))
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=20)
    willr = calculate_williams_r(high, low, close, period=14)
    
    # Volume ratio (current vs 20-bar MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1)
    
    # Signals array
    signals = np.zeros(n)
    SIZE = 0.28  # Position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 60  # Donchian(20) + Williams %R(14) need ~20 bars
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(willr[i]) or np.isnan(donchian_upper[i]):
            signals[i] = 0.0
            continue
        
        # === HTF TREND ===
        htf_bull_trend = htf_bull_aligned[i] > 0.5 if not np.isnan(htf_bull_aligned[i]) else False
        htf_bear_trend = htf_bear_aligned[i] > 0.5 if not np.isnan(htf_bear_aligned[i]) else False
        
        # === DONCHIAN BREAKOUT SIGNALS ===
        # Price breaks above 20-bar high = bullish breakout
        donchian_break_up = close[i] > donchian_upper[i] and close[i-1] <= donchian_upper[i-1]
        # Price breaks below 20-bar low = bearish breakout
        donchian_break_down = close[i] < donchian_lower[i] and close[i-1] >= donchian_lower[i-1]
        
        # === WILLIAMS %R SIGNALS ===
        # %R crosses above -20 = bullish momentum (recovering from oversold)
        willr_bull_cross = willr[i] > -20 and willr[i-1] <= -20
        # %R crosses below -80 = bearish momentum (breaking into oversold)
        willr_bear_cross = willr[i] < -80 and willr[i-1] >= -80
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG ENTRY: %R bullish cross + Donchian breakout + volume + HTF bull/neutral
            # Only require %R cross OR Donchian break (not both) to avoid over-filtering
            bull_momentum = willr_bull_cross or donchian_break_up
            
            if bull_momentum and vol_spike:
                # Prefer HTF bull, but allow neutral for more trades
                if htf_bull_trend:
                    desired_signal = SIZE
                elif not htf_bear_trend:  # Neutral
                    desired_signal = SIZE
            
            # SHORT ENTRY: %R bearish cross + Donchian breakdown + volume + HTF bear/neutral
            bear_momentum = willr_bear_cross or donchian_break_down
            
            if bear_momentum and vol_spike:
                if htf_bear_trend:
                    desired_signal = -SIZE
                elif not htf_bull_trend:  # Neutral
                    desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # ATR trailing stop (2.5x ATR)
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                
                # Exit if momentum breaks (willr crosses below -50 = weakening)
                if willr[i] < -50 and willr[i-1] >= -50:
                    desired_signal = 0.0
                
                # Exit if HTF turns bearish
                if htf_bear_trend:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # ATR trailing stop
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                
                # Exit if momentum breaks (willr crosses above -50 = weakening)
                if willr[i] > -50 and willr[i-1] <= -50:
                    desired_signal = 0.0
                
                # Exit if HTF turns bullish
                if htf_bull_trend:
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