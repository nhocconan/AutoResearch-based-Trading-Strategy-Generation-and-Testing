#!/usr/bin/env python3
"""
Experiment #025: 4h RSI Extreme + Volume Spike + Daily SMA200

HYPOTHESIS: RSI extremes (30/70) capture high-probability reversal zones.
Volume spike confirms the move has conviction. Daily SMA200 filters countertrend.
4h timeframe gives enough bars for meaningful sample size while avoiding fee drag.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: RSI<30 + volume spike + above SMA200 = oversold bounce (50-100% gains common)
- Bear: RSI>70 + volume spike + below SMA200 = distribution/reversal (captures tops)
- Mean reversion works in both directions - not biased to bull or bear

EXPECTED TRADES: 150-250 total over 4 years (37-62/year per symbol)
- RSI(14) extreme triggers: ~50-100/year per symbol
- Volume spike filter (1.5x): reduces by ~30%
- SMA200 trend filter: reduces by ~20%
- Final: ~150-250 trades = statistical validity
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_extreme_vol_sma200_v1"
timeframe = "4h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(prices, period=14):
    """RSI with min_periods"""
    close = pd.Series(prices)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1d = get_htf_data(prices, '1d')
    
    # Daily SMA200
    sma200_1d = pd.Series(df_1d['close'].values).rolling(window=200, min_periods=200).mean().values
    sma200_aligned = align_htf_to_ltf(prices, df_1d, sma200_1d)
    
    # === Local 4h indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    
    warmup = 250  # Enough for RSI(14), ATR(14), SMA200(1d)
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(sma200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === TREND DIRECTION ===
        above_daily_sma = close[i] > sma200_aligned[i]
        below_daily_sma = close[i] < sma200_aligned[i]
        
        # === RSI EXTREME CONDITIONS ===
        rsi_oversold = rsi_14[i] < 30
        rsi_overbought = rsi_14[i] > 70
        
        # RSI neutralization for exit
        rsi_neutral_long = rsi_14[i] > 55  # Exit long when RSI recovers
        rsi_neutral_short = rsi_14[i] < 45  # Exit short when RSI drops
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: RSI oversold + volume spike + above daily SMA
            if rsi_oversold and vol_spike and above_daily_sma:
                desired_signal = SIZE
            
            # SHORT: RSI overbought + volume spike + below daily SMA
            elif rsi_overbought and vol_spike and below_daily_sma:
                desired_signal = -SIZE
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Trailing high
                if i == entry_bar or high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 2.5 ATR from highest
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if RSI neutralizes (mean reversion complete)
                elif rsi_neutral_long:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Trailing low
                if i == entry_bar or low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 2.5 ATR from lowest
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if RSI neutralizes
                elif rsi_neutral_short:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 2 bars to reduce fee churn ===
        if in_position and (i - entry_bar) < 2:
            desired_signal = position_side * SIZE
        
        # === EXECUTE NEW POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
        
        signals[i] = desired_signal
    
    return signals