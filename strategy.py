#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian Breakout + Weekly EMA Trend + Volume + ADX (1d)

HYPOTHESIS: Daily timeframe with weekly trend filter captures multi-week trends
while avoiding overtrading. Weekly EMA21 determines bull/bear regime.
Donchian(20) breakout signals momentum entries. ADX confirms trend strength.

WHY IT SHOULD WORK IN BOTH MARKETS:
- Bull: Breakout above 20d high + above weekly EMA + ADX>25 = strong momentum continuation
- Bear: Breakdown below 20d low + below weekly EMA + ADX>25 = strong short
- Weekly EMA smooths noise, ADX prevents choppy market entries

EXPECTED TRADES: 40-80 total over 4 years (10-20/year)
- Donchian(20) on 1d = break every 20-40 bars = 91-182 potential/year
- Volume spike filter → reduces by ~40%
- Weekly EMA trend filter → reduces by ~30%
- ADX>25 filter → reduces by ~25%
- Final: ~40-80 trades = statistical validity with 1d timeframe
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_ema_vol_adx_v1"
timeframe = "1d"
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

def calculate_adx(high, low, close, period=14):
    """ADX (Average Directional Index)"""
    n = len(close)
    if n < period * 2:
        return np.full(n, np.nan)
    
    # True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    
    # Smoothed values
    atr_smooth = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    # DI
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
    
    # DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # ADX
    adx = pd.Series(dx).ewm(span=period, min_periods=period * 2, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(21) for trend
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=21, min_periods=21, adjust=False).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # === Local 1d indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Donchian Channel(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20 bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1)
    
    # === Signals ===
    signals = np.zeros(n)
    SIZE = 0.25
    HALF_SIZE = SIZE / 2
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    trailing_high = 0.0
    trailing_low = 0.0
    takeprofit_hit = False
    
    warmup = 60  # Enough for Donchian20, ATR14, ADX, VWAP
    
    for i in range(warmup, n):
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(weekly_ema_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(adx_14[i]):
            adx_val = 0.0
        else:
            adx_val = adx_14[i]
        
        # === TREND DIRECTION: Weekly EMA21 ===
        bull_trend = close[i] > weekly_ema_aligned[i]
        bear_trend = close[i] < weekly_ema_aligned[i]
        
        # === ADX REGIME: Trend must be present ===
        trend_present = adx_val > 22.0
        
        # === VOLUME CONFIRMATION ===
        vol_spike = vol_ratio[i] > 1.5
        
        # === DONCHIAN BREAKOUT ===
        prev_donchian_high = donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else np.nan
        prev_donchian_low = donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else np.nan
        
        bullish_breakout = (not np.isnan(prev_donchian_high) and 
                           high[i] > prev_donchian_high)
        bearish_breakout = (not np.isnan(prev_donchian_low) and 
                           low[i] < prev_donchian_low)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        if not in_position:
            # LONG: Bullish breakout + volume spike + bull trend + ADX
            if bullish_breakout and vol_spike and bull_trend and trend_present:
                desired_signal = SIZE
                in_position = True
                position_side = 1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
                takeprofit_hit = False
            
            # SHORT: Bearish breakout + volume spike + bear trend + ADX
            elif bearish_breakout and vol_spike and bear_trend and trend_present:
                desired_signal = -SIZE
                in_position = True
                position_side = -1
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                trailing_high = high[i]
                trailing_low = low[i]
                takeprofit_hit = False
        
        # === EXIT LOGIC ===
        if in_position:
            if position_side > 0:
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Calculate PnL in ATR terms
                pnl_atr = (close[i] - entry_price) / entry_atr
                
                # Take profit at 2R: reduce to half
                if not takeprofit_hit and pnl_atr >= 2.0:
                    desired_signal = HALF_SIZE
                    takeprofit_hit = True
                elif takeprofit_hit:
                    desired_signal = HALF_SIZE
                else:
                    desired_signal = SIZE
                
                # Stop loss: 2.5 ATR from highest
                stop_price = trailing_high - 2.5 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips
                elif close[i] < weekly_ema_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Calculate PnL in ATR terms
                pnl_atr = (entry_price - close[i]) / entry_atr
                
                # Take profit at 2R: reduce to half
                if not takeprofit_hit and pnl_atr >= 2.0:
                    desired_signal = -HALF_SIZE
                    takeprofit_hit = True
                elif takeprofit_hit:
                    desired_signal = -HALF_SIZE
                else:
                    desired_signal = -SIZE
                
                # Stop loss: 2.5 ATR from lowest
                stop_price = trailing_low + 2.5 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if trend flips
                elif close[i] > weekly_ema_aligned[i]:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        signals[i] = desired_signal
    
    return signals