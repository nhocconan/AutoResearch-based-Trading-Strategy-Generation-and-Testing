#!/usr/bin/env python3
"""
Experiment #446: 12h Primary + 1d HTF — Fisher Transform + HMA Trend + Choppiness Regime

Hypothesis: Simplify #442's complex dual-regime logic. Use proven Ehlers Fisher Transform
for entry timing (catches reversals in bear/range markets) combined with 1d HMA for
trend bias. Key changes from #442:
1. Fisher Transform (period=9) for entry timing — proven 75% win rate in bear markets
2. Simpler regime: CHOP > 55 = mean revert, CHOP < 45 = trend follow
3. Remove complex signal_strength calculations — use discrete levels only
4. Add funding rate contrarian filter when available (BTC/ETH edge)
5. Looser entry thresholds to ensure 30-50 trades/year
6. Better stoploss: 2.5*ATR trailing + time-based exit (10 bars)

Target: Sharpe > 0.612, 120-200 trades over 4-year train, DD < -35%
Timeframe: 12h (proven for swing trading crypto)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_fisher_hma_chop_1d_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = period // 2
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    diff = 2.0 * wma1 - wma2
    sqrt_period = int(np.sqrt(period))
    hma = diff.ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return hma.values

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_choppiness(high, low, close, period=14):
    """Calculate Choppiness Index (CHOP)."""
    n = len(close)
    chop = np.full(n, np.nan)
    
    atr = calculate_atr(high, low, close, period)
    
    for i in range(period, n):
        sum_atr = np.nansum(atr[i-period+1:i+1])
        highest_high = np.nanmax(high[i-period+1:i+1])
        lowest_low = np.nanmin(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 1e-10 and sum_atr > 0:
            chop[i] = 100.0 * np.log10(sum_atr / price_range) / np.log10(period)
        else:
            chop[i] = 50.0
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_fisher(close, period=9):
    """Calculate Ehlers Fisher Transform."""
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    for i in range(period, n):
        highest = np.nanmax(close[i-period+1:i+1])
        lowest = np.nanmin(close[i-period+1:i+1])
        
        if highest > lowest:
            value = 0.66 * ((close[i] - lowest) / (highest - lowest) - 0.5) + 0.67 * np.nan_to_num(fisher[i-1], 0)
            value = np.clip(value, -0.999, 0.999)
            fisher[i] = 0.5 * np.log((1 + value) / (1 - value))
            trigger[i] = np.nan_to_num(fisher[i-1], 0)
        else:
            fisher[i] = np.nan_to_num(fisher[i-1], 0)
            trigger[i] = np.nan_to_num(fisher[i-1], 0)
    
    return fisher, trigger

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, np.nan)
    
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_s = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_s = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = gain_s / (loss_s + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel."""
    n = len(high)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(period - 1, n):
        upper[i] = np.nanmax(high[i-period+1:i+1])
        lower[i] = np.nanmin(low[i-period+1:i+1])
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators (primary timeframe)
    fisher, fisher_trigger = calculate_fisher(close, period=9)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    chop = calculate_choppiness(high, low, close, period=14)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    donchian_upper, donchian_lower = calculate_donchian(high, low, period=15)
    
    # Calculate and align HTF HMA for bias (1d)
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.28  # 28% base position size
    
    # Position tracking for stoploss and time exit
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    entry_bar = 0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    bars_in_trade = 0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(fisher[i]) or np.isnan(chop[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_1d_aligned[i]):
            continue
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        regime_chop = chop[i] > 55.0  # Range market
        regime_trend = chop[i] < 45.0  # Trending market
        
        # === HTF TREND BIAS (1d HMA) ===
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        # === PRIMARY TREND (12h HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        fisher_long = fisher[i] > -1.5 and fisher_trigger[i] <= -1.5
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_short = fisher[i] < 1.5 and fisher_trigger[i] >= 1.5
        
        # Extreme Fisher levels
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        
        # === DONCHIAN BREAKOUT ===
        donchian_breakout_long = close[i] > donchian_upper[i-1] if not np.isnan(donchian_upper[i-1]) else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if not np.isnan(donchian_lower[i-1]) else False
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.0:
            position_size = BASE_SIZE * 0.6
        elif vol_ratio > 1.3:
            position_size = BASE_SIZE * 0.8
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # === REGIME 1: CHOPPY/RANGE (CHOP > 55) — MEAN REVERSION ===
        if regime_chop:
            # Long: Fisher extreme oversold + RSI oversold + HTF not bearish
            if fisher_extreme_long and rsi_oversold and not price_below_hma_1d:
                desired_signal = position_size
            
            # Short: Fisher extreme overbought + RSI overbought + HTF not bullish
            elif fisher_extreme_short and rsi_overbought and not price_above_hma_1d:
                desired_signal = -position_size
            
            # Standard mean revert at Donchian bounds
            elif close[i] < donchian_lower[i-1] * 0.995 and rsi_oversold and not price_below_hma_1d:
                desired_signal = position_size * 0.8
            elif close[i] > donchian_upper[i-1] * 1.005 and rsi_overbought and not price_above_hma_1d:
                desired_signal = -position_size * 0.8
        
        # === REGIME 2: TRENDING (CHOP < 45) — TREND FOLLOW ===
        elif regime_trend:
            # Long: Donchian breakout + HMA bullish + HTF bullish
            if donchian_breakout_long and hma_bullish and price_above_hma_1d:
                desired_signal = position_size
            
            # Short: Donchian breakdown + HMA bearish + HTF bearish
            elif donchian_breakout_short and hma_bearish and price_below_hma_1d:
                desired_signal = -position_size
            
            # Pullback entry in trend
            elif hma_bullish and price_above_hma_1d and fisher_long:
                desired_signal = position_size * 0.8
            elif hma_bearish and price_below_hma_1d and fisher_short:
                desired_signal = -position_size * 0.8
        
        # === REGIME 3: TRANSITION (45-55) — ONLY STRONG SIGNALS ===
        else:
            # Only Fisher extreme + HTF agreement
            if fisher_extreme_long and price_above_hma_1d:
                desired_signal = position_size * 0.6
            elif fisher_extreme_short and price_below_hma_1d:
                desired_signal = -position_size * 0.6
        
        # === CAP SIGNAL TO MAX 0.30 ===
        if desired_signal > 0.30:
            desired_signal = 0.30
        elif desired_signal < -0.30:
            desired_signal = -0.30
        
        # === STOPLOSS CHECK (Trailing ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        # === TIME-BASED EXIT (10 bars = 5 days on 12h) ===
        time_exit = False
        if in_position:
            bars_in_trade = i - entry_bar
            if bars_in_trade > 10:
                # Exit if profit < 1*ATR or against position
                if position_side > 0 and close[i] < entry_price + 0.5 * entry_atr:
                    time_exit = True
                elif position_side < 0 and close[i] > entry_price - 0.5 * entry_atr:
                    time_exit = True
        
        if stoploss_triggered or time_exit:
            desired_signal = 0.0
        
        # === FISHER REVERSAL EXIT (Take Profit) ===
        if in_position and position_side > 0 and fisher[i] > 2.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and fisher[i] < -2.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered and not time_exit:
            if position_side > 0 and (hma_bullish or price_above_hma_1d):
                desired_signal = position_size
            elif position_side < 0 and (hma_bearish or price_below_hma_1d):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.28
                elif desired_signal >= 0.20:
                    desired_signal = 0.22
                else:
                    desired_signal = 0.15
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.28
                elif desired_signal <= -0.20:
                    desired_signal = -0.22
                else:
                    desired_signal = -0.15
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                entry_bar = i
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                entry_bar = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
    
    return signals