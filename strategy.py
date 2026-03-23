#!/usr/bin/env python3
"""
Experiment #350: 1h Primary + 4h/12h HTF — Fisher Transform Reversal with Regime Filter

Hypothesis: Previous 1h strategies failed because:
1. CRSI thresholds too strict (RSI<10 or >90 rarely triggers)
2. Too many confluence filters required simultaneously (all must agree)
3. No proper regime adaptation for bear/range markets (2025 test period)

This strategy uses Ehrler's Fisher Transform for entry timing (proven in bear markets):
1. 12h HMA(21) as MACRO BIAS (only long if price > 12h HMA, only short if price < 12h HMA)
2. 4h Choppiness Index for regime detection (CHOP>55=range mean-revert, CHOP<45=trend follow)
3. RANGE REGIME: 1h Fisher Transform extremes for reversals (Fisher<-1.5 long, Fisher>+1.5 short)
4. TREND REGIME: 1h HMA(16/48) crossover + 4h ADX>20 for trend confirmation
5. Session filter: Only trade 8-20 UTC (high volume periods)
6. Volume confirmation: volume > 0.7x 20-period average (not too strict)
7. ATR(14) trailing stop at 2.5x for risk management

KEY INSIGHT: Fisher Transform catches reversals better than RSI in bear/range markets.
The 12h HMA bias is more responsive than 1d for 1h timeframe strategies.
Relaxed thresholds (Fisher ±1.5, RSI 35/65, volume 0.7x) ensure 30-80 trades/year.

TARGET: 40-80 trades/year on 1h, Sharpe > 0.4 on ALL symbols (BTC/ETH/SOL)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_fisher_regime_4h12h_session_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    wma_half = close_s.ewm(span=period//2, min_periods=period//2, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma_diff = 2 * wma_half - wma_full
    hma = wma_diff.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
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

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / (atr + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    atr = calculate_atr(high, low, close, period)
    
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    sum_atr = pd.Series(atr).rolling(window=period, min_periods=period).sum().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / (highest_high - lowest_low + 1e-10)) / np.log10(period)
    
    chop = np.nan_to_num(chop, nan=50.0)
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0).values

def calculate_fisher_transform(high, low, close, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Converts price to a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, Short when crosses below +1.5
    """
    # Calculate typical price
    typical = (high + low + close) / 3.0
    typical_s = pd.Series(typical)
    
    # Normalize price to range -1 to +1
    highest = typical_s.rolling(window=period, min_periods=period).max().values
    lowest = typical_s.rolling(window=period, min_periods=period).min().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = 2 * ((typical - lowest) / (highest - lowest + 1e-10)) - 1
        normalized = np.clip(normalized, -0.99, 0.99)  # Prevent log domain errors
    
    # Fisher Transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized + 1e-10))
    
    # Signal line (1-period lag of Fisher)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs moving average."""
    vol_s = pd.Series(volume)
    vol_ma = vol_s.rolling(window=period, min_periods=period).mean().values
    with np.errstate(divide='ignore', invalid='ignore'):
        vol_ratio = volume / (vol_ma + 1e-10)
    return vol_ratio

def get_hour_from_open_time(open_time_arr):
    """Extract UTC hour from open_time (milliseconds timestamp)."""
    # open_time is in milliseconds
    hours = (open_time_arr // (1000 * 3600)) % 24
    return hours

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    open_time = prices["open_time"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 1h indicators (primary timeframe)
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    fisher, fisher_signal = calculate_fisher_transform(high, low, close, period=9)
    vol_ratio = calculate_volume_ratio(volume, period=20)
    
    # HMA for trend detection (fast and slow)
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    
    # Calculate 4h Choppiness and align to 1h
    chop_4h_raw = calculate_choppiness(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    chop_4h_aligned = align_htf_to_ltf(prices, df_4h, chop_4h_raw)
    
    # Calculate 4h ADX and align to 1h
    adx_4h_raw = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, period=14)
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h_raw)
    
    # Calculate and align 12h HMA for macro bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Extract UTC hour for session filter
    hours = get_hour_from_open_time(open_time)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% position size for 1h (target 40-80 trades/year)
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track Fisher crossovers for cleaner entries
    prev_fisher = fisher[0]
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            continue
        if np.isnan(rsi_14[i]) or np.isnan(fisher[i]) or np.isnan(vol_ratio[i]):
            signals[i] = 0.0
            continue
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            continue
        if np.isnan(chop_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # === SESSION FILTER (8-20 UTC only) ===
        in_session = 8 <= hours[i] <= 20
        
        # === VOLUME FILTER (>0.7x average) ===
        volume_ok = vol_ratio[i] > 0.7
        
        # === MACRO BIAS (12h HMA - HARD FILTER) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (4h Choppiness) ===
        is_choppy = chop_4h_aligned[i] > 55.0  # High choppiness = range regime (mean revert)
        is_trending = chop_4h_aligned[i] < 45.0  # Low choppiness = trend regime (breakout)
        
        # === FISHER TRANSFORM CROSSOVER DETECTION ===
        fisher_cross_up = (prev_fisher < -1.5) and (fisher[i] >= -1.5)
        fisher_cross_down = (prev_fisher > 1.5) and (fisher[i] <= 1.5)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Confluence counter (need at least 2 of 3 filters)
        if in_session and volume_ok:
            if is_choppy:
                # RANGE REGIME: Fisher mean reversion
                # Need: Fisher crossover + 12h HMA bias + (RSI confirmation OR volume)
                
                confluence_long = 0
                confluence_short = 0
                
                if fisher_cross_up:
                    confluence_long += 1
                if price_above_hma_12h:
                    confluence_long += 1
                if rsi_14[i] < 45:
                    confluence_long += 1
                
                if fisher_cross_down:
                    confluence_short += 1
                if price_below_hma_12h:
                    confluence_short += 1
                if rsi_14[i] > 55:
                    confluence_short += 1
                
                if confluence_long >= 2:
                    desired_signal = BASE_SIZE
                elif confluence_short >= 2:
                    desired_signal = -BASE_SIZE
            
            elif is_trending:
                # TREND REGIME: HMA crossover + ADX confirmation
                # Need: HMA crossover + 12h HMA bias + ADX
                
                hma_bullish = hma_16[i] > hma_48[i]
                hma_bearish = hma_16[i] < hma_48[i]
                trend_strong = adx_4h_aligned[i] > 20.0
                
                confluence_long = 0
                confluence_short = 0
                
                if hma_bullish:
                    confluence_long += 1
                if price_above_hma_12h:
                    confluence_long += 1
                if trend_strong:
                    confluence_long += 1
                
                if hma_bearish:
                    confluence_short += 1
                if price_below_hma_12h:
                    confluence_short += 1
                if trend_strong:
                    confluence_short += 1
                
                if confluence_long >= 2:
                    desired_signal = BASE_SIZE
                elif confluence_short >= 2:
                    desired_signal = -BASE_SIZE
            
            else:
                # NEUTRAL REGIME (45 <= CHOP <= 55): Only Fisher extremes
                if fisher[i] < -1.8 and price_above_hma_12h:
                    desired_signal = BASE_SIZE * 0.8
                elif fisher[i] > 1.8 and price_below_hma_12h:
                    desired_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
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
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXIT (mean reversion complete) ===
        if in_position and position_side > 0 and rsi_14[i] > 65:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 35:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position unless clear exit trigger ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0:
                # Hold long if bias still valid
                if price_above_hma_12h and rsi_14[i] < 65:
                    desired_signal = BASE_SIZE
            elif position_side < 0:
                # Hold short if bias still valid
                if price_below_hma_12h and rsi_14[i] > 35:
                    desired_signal = -BASE_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                # Position flip
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = desired_signal
        prev_fisher = fisher[i]
    
    return signals