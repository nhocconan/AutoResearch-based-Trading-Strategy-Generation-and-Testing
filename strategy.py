#!/usr/bin/env python3
"""
Experiment #021: 4h TRIX + EMA21 Cross + Donchian + Volume

HYPOTHESIS: Combine the BEST elements from DB winners and session best:
1. TRIX momentum (confirmed in mtf_4h_trix_donchian_chop_12h_v1: Sharpe=0.362)
2. EMA21 cross for signal confirmation (simpler than complex filters)
3. Donchian(20) breakout as price structure entry
4. Volume confirmation (1.5x)
5. 12h HTF EMA21 for trend bias

WHY IT SHOULD WORK IN BULL + BEAR + RANGE:
- Bull: TRIX crosses positive + price above EMA21 + above Donchian high = strong long
- Bear: TRIX crosses negative + price below EMA21 + below Donchian low = strong short  
- Range: TRIX oscillates around zero = no entries (avoids whipsaws)
- TRIX smooths noise better than raw momentum indicators

DIFFERENCE FROM #020:
- #020: TRIX(20) + Donchian(20) + Choppiness + 12h ref = 160 trades, Sharpe 0.362
- #021: TRIX(14) + EMA21 cross + Donchian(20) + 12h ref = different signal structure
- Simpler: EMA21 cross replaces Choppiness filter (avoids missing moves)

TARGET: 100-200 total trades over 4 years (25-50/year)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_trix_ema_cross_donchian_vol_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_trix(close, period=14):
    """
    TRIX - Triple EMA Rate of Change
    - Positive TRIX = bullish momentum
    - Negative TRIX = bearish momentum
    - Crosses around zero = momentum shift
    """
    n = len(close)
    if n < period * 3:
        return np.full(n, np.nan)
    
    # Triple EMA
    ema1 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=period, min_periods=period, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # Rate of change of triple EMA
    trix = np.full(n, np.nan)
    for i in range(period * 3, n):
        if ema3[i - period] != 0:
            trix[i] = 100 * ((ema3[i] - ema3[i - period]) / ema3[i - period])
    
    return trix

def calculate_donchian(high, low, period=20):
    """
    Donchian Channel
    - Upper = highest high over period
    - Lower = lowest low over period
    """
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_12h = get_htf_data(prices, '12h')
    
    # 12h EMA(21) for HTF trend direction
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    htf_ema_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # === Local 4h indicators ===
    # TRIX(14) for momentum
    trix = calculate_trix(close, period=14)
    
    # TRIX signal line (EMA of TRIX, period 9)
    trix_series = pd.Series(trix)
    trix_signal = trix_series.ewm(span=9, min_periods=9, adjust=False).mean().values
    
    # EMA(21) for trend
    ema_21 = pd.Series(close).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Donchian(20) for breakout
    donchian_up, donchian_lo = calculate_donchian(high, low, period=20)
    
    # ATR(14) for stoploss
    tr = np.maximum(high - low, np.maximum(abs(high - np.roll(close, 1)), abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Volume ratio (20-period MA)
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
    
    warmup = 200  # 20 (Donchian) + 42 (TRIX*3) + 20 (vol MA) + some buffer
    
    for i in range(warmup, n):
        # Check data availability
        if np.isnan(trix[i]) or np.isnan(trix_signal[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(ema_21[i]) or np.isnan(htf_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_up[i]) or np.isnan(donchian_lo[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # === HTF TREND (12h EMA21) ===
        htf_trend_up = close[i] > htf_ema_aligned[i]
        htf_trend_down = close[i] < htf_ema_aligned[i]
        
        # === LOCAL TREND (4h EMA21) ===
        local_trend_up = close[i] > ema_21[i]
        local_trend_down = close[i] < ema_21[i]
        
        # === TRIX CROSS (momentum shift) ===
        trix_bullish_cross = (trix[i - 1] <= trix_signal[i - 1]) and (trix[i] > trix_signal[i])
        trix_bearish_cross = (trix[i - 1] >= trix_signal[i - 1]) and (trix[i] < trix_signal[i])
        
        # TRIX momentum strength
        trix_positive = trix[i] > 0
        trix_negative = trix[i] < 0
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_up = donchian_up[i - 1]
        prev_donchian_lo = donchian_lo[i - 1]
        breakout_up = close[i] > prev_donchian_up
        breakout_down = close[i] < prev_donchian_lo
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # === LONG: Momentum cross up + local trend up + HTF trend up + breakout + volume ===
            # Option 1: Strong signal (all conditions)
            if trix_bullish_cross and local_trend_up and htf_trend_up and vol_spike:
                if breakout_up:
                    desired_signal = SIZE
                else:
                    # Still enter on momentum cross with trend confirmation
                    desired_signal = SIZE * 0.8  # Smaller size without breakout
            
            # === SHORT: Momentum cross down + local trend down + HTF trend down + breakout + volume ===
            if trix_bearish_cross and local_trend_down and htf_trend_down and vol_spike:
                if breakout_down:
                    desired_signal = -SIZE
                else:
                    desired_signal = -SIZE * 0.8
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Exit on TRIX bearish cross (momentum flip)
                if trix_bearish_cross:
                    desired_signal = 0.0
                
                # Exit on HTF trend flip
                if htf_trend_down:
                    desired_signal = 0.0
                
                # Trailing stop: 2.5 ATR from entry
                stop_price = entry_price - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
            
            elif position_side < 0:
                # Exit on TRIX bullish cross (momentum flip)
                if trix_bullish_cross:
                    desired_signal = 0.0
                
                # Exit on HTF trend flip
                if htf_trend_up:
                    desired_signal = 0.0
                
                # Trailing stop: 2.5 ATR from entry
                stop_price = entry_price + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
        
        # === MINIMUM HOLD: 4 bars to avoid fee churn ===
        if in_position and (i - entry_bar) < 4:
            desired_signal = position_side * SIZE
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                # New position or flip
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
        
        signals[i] = desired_signal
    
    return signals