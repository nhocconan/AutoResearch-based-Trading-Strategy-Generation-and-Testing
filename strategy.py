#!/usr/bin/env python3
"""
Experiment #023: Weekly Donchian + Williams %R + Volume Confirmation (1d)

HYPOTHESIS: Weekly Donchian(13) breakout with Williams %R confirmation and 
volume spike generates high-quality, low-frequency trades on daily timeframe.

WHY IT SHOULD WORK:
- Weekly Donchian(13) = 65 trading days = ~13 breaks/year (5-8 valid after filters)
- Williams %R(14) confirms momentum: < -80 for long, > -20 for short
- Volume spike (1.8x) confirms institutional involvement
- Weekly EMA(50) provides trend direction filter
- 1d primary + 1w HTF = very low trade frequency = minimal fee drag

TRADE COUNT TARGET: 35-60 total over 4 years (8-15/year)
- 13 weekly Donchian breaks/year × ~40% volume confirmation = ~5 valid
- Williams %R filter adds ~20% reduction = ~4 trades/year
- Result: 16-20 trades × 2 symbols = 32-40 total (in range)

RISK MANAGEMENT:
- Stoploss: 3 ATR from entry (wide enough to survive volatility)
- Max position: 30%
- Minimum hold: 5 days (reduce fee churn)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_williams_1w_vol_v1"
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

def calculate_donchian(high, low, period=20):
    """Donchian Channel - highest high and lowest low over period"""
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def calculate_williams_r(high, low, close, period=14):
    """Williams %R - momentum oscillator"""
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max()
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min()
    
    willr = pd.Series(index=range(len(close)), dtype=float)
    for i in range(len(close)):
        if not np.isnan(highest_high.iloc[i]) and not np.isnan(lowest_low.iloc[i]):
            hh = highest_high.iloc[i]
            ll = lowest_low.iloc[i]
            if hh != ll:
                willr.iloc[i] = -100 * (hh - close[i]) / (hh - ll)
            else:
                willr.iloc[i] = -50
        else:
            willr.iloc[i] = np.nan
    
    return willr.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load Weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # === Weekly (1w) Indicators ===
    # Weekly EMA(50) for trend direction
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Weekly Donchian(13) for breakout structure
    donchian_upper_1w, donchian_lower_1w = calculate_donchian(
        df_1w['high'].values, df_1w['low'].values, period=13
    )
    # Align to daily
    donchian_upper_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_upper_1w)
    donchian_lower_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_lower_1w)
    
    # Weekly Williams %R(14) for momentum confirmation
    williams_1w = calculate_williams_r(
        df_1w['high'].values, df_1w['low'].values, df_1w['close'].values, period=14
    )
    williams_1w_aligned = align_htf_to_ltf(prices, df_1w, williams_1w)
    
    # Weekly volume average for spike detection
    vol_1w_avg = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=10).mean().values
    vol_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_1w_avg)
    
    # === Daily Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Daily EMA(20) for local trend
    ema20_1d = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Daily Williams %R(14) for entry timing
    williams_1d = calculate_williams_r(high, low, close, period=14)
    
    # Daily volume ratio
    vol_30d_avg = pd.Series(volume).rolling(window=30, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_30d_avg > 0, vol_30d_avg, 1)
    
    # === Signals ===
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
    
    warmup = 200  # EMA(50) weekly + Williams(14) + volume
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        
        # Get aligned weekly values
        weekly_ema = ema50_1w_aligned[i]
        weekly_upper = donchian_upper_1w_aligned[i]
        weekly_lower = donchian_lower_1w_aligned[i]
        weekly_willr = williams_1w_aligned[i]
        weekly_vol_avg = vol_1w_aligned[i]
        
        if np.isnan(weekly_ema) or np.isnan(weekly_upper):
            signals[i] = 0.0
            continue
        
        # Current position in weekly data
        current_week_close = df_1w['close'].values[-1] if len(df_1w) > 0 else close[i]
        
        # === ENTRY CONDITIONS ===
        desired_signal = 0.0
        
        # Volume spike: 1.8x average daily volume
        vol_spike = vol_ratio[i] > 1.8
        
        # Weekly trend: bull if close > weekly EMA, bear if close < weekly EMA
        weekly_bull = close[i] > weekly_ema if not np.isnan(weekly_ema) else False
        weekly_bear = close[i] < weekly_ema if not np.isnan(weekly_ema) else False
        
        # Weekly Williams %R momentum: < -80 = oversold (long), > -20 = overbought (short)
        weekly_oversold = weekly_willr < -80 if not np.isnan(weekly_willr) else False
        weekly_overbought = weekly_willr > -20 if not np.isnan(weekly_willr) else False
        
        # Daily Williams %R for confirmation
        daily_willr = williams_1d[i]
        daily_oversold = daily_willr < -80 if not np.isnan(daily_willr) else False
        daily_overbought = daily_willr > -20 if not np.isnan(daily_willr) else False
        
        # === LONG ENTRY ===
        if not in_position:
            # Weekly breakout: close above weekly Donchian high
            weekly_breakout_bull = close[i] > weekly_upper if not np.isnan(weekly_upper) else False
            
            # Entry: weekly breakout + volume spike + oversold Williams + bull trend
            if weekly_breakout_bull and vol_spike and (weekly_oversold or daily_oversold) and weekly_bull:
                desired_signal = SIZE
                
            # === SHORT ENTRY ===
            # Weekly breakdown: close below weekly Donchian low
            weekly_breakout_bear = close[i] < weekly_lower if not np.isnan(weekly_lower) else False
            
            if weekly_breakout_bear and vol_spike and (weekly_overbought or daily_overbought) and weekly_bear:
                desired_signal = -SIZE
        
        # === STOPLOSS AND EXIT ===
        if in_position:
            if position_side > 0:  # Long position
                # Update trailing high
                if high[i] > trailing_high:
                    trailing_high = high[i]
                
                # Stop: 3 ATR from entry
                stop_price = entry_price - 3.0 * entry_atr
                if low[i] < stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price crosses below daily EMA(20) - local reversal
                if close[i] < ema20_1d[i] and not np.isnan(ema20_1d[i]):
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if Williams %R reaches overbought (> -20)
                if daily_willr > -20 and not np.isnan(daily_willr):
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                    
            elif position_side < 0:  # Short position
                # Update trailing low
                if low[i] < trailing_low:
                    trailing_low = low[i]
                
                # Stop: 3 ATR from entry
                stop_price = entry_price + 3.0 * entry_atr
                if high[i] > stop_price:
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if price crosses above daily EMA(20) - local reversal
                if close[i] > ema20_1d[i] and not np.isnan(ema20_1d[i]):
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
                
                # Exit if Williams %R reaches oversold (< -80)
                if daily_willr < -80 and not np.isnan(daily_willr):
                    desired_signal = 0.0
                    in_position = False
                    position_side = 0
        
        # === MINIMUM HOLD: 5 days to avoid fee churn ===
        if in_position and (i - entry_bar) < 5:
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
        
        signals[i] = desired_signal
    
    return signals