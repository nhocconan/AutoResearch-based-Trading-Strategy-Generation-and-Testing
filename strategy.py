#!/usr/bin/env python3
"""
Experiment #454: 4h Primary + 12h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After #451 failed (Sharpe=-0.080), simplify the logic. The current best
(Sharpe=0.612) uses HMA trend + RSI pullback. Key changes:
1. Remove complex Fisher/CRSI/Choppiness regime switching (too many filters = 0 trades)
2. Use ADX + HMA for simpler regime detection (ADX>25=trend, <20=range)
3. RSI(14) pullback entries in HTF trend direction (proven pattern)
4. 12h + 1d HMA for stronger bias confirmation
5. ATR(14) trailing stop at 2.5x
6. Position size: 0.25 base, 0.30 on strong confluence, discrete levels
7. Ensure 40-80 trades over 4-year train (10-20/year)

Target: Sharpe > 0.612, DD < -35%, trades >= 40 on train
Timeframe: 4h (proven best for swing trading crypto)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h1d_v1"
timeframe = "4h"
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

def calculate_rsi(close, period):
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
    n = len(close)
    adx = np.full(n, np.nan)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di_s = pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    minus_di_s = pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = 100.0 * plus_di_s / (atr + 1e-10)
        minus_di = 100.0 * minus_di_s / (atr + 1e-10)
        dx = 100.0 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    taker_buy_volume = prices["taker_buy_volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h indicators (primary timeframe)
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    atr_14 = calculate_atr(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    
    # Calculate taker buy ratio for volume confirmation
    taker_ratio = np.zeros(n)
    for i in range(n):
        if volume[i] > 0:
            taker_ratio[i] = taker_buy_volume[i] / volume[i]
        else:
            taker_ratio[i] = 0.5
    
    # Calculate and align HTF HMA for bias (12h and 1d)
    hma_12h_raw = calculate_hma(df_12h['close'].values, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate median ATR for vol filter
    valid_atr = atr_14[200:]
    atr_median = np.nanmedian(valid_atr[~np.isnan(valid_atr)])
    if np.isnan(atr_median) or atr_median <= 0:
        atr_median = np.nanmean(valid_atr[~np.isnan(valid_atr)])
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25  # 25% base position size for 4h
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            continue
        if np.isnan(rsi_14[i]) or np.isnan(adx_14[i]):
            continue
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            continue
        
        # === REGIME DETECTION (ADX) ===
        regime_trend = adx_14[i] > 25.0  # Trending
        regime_range = adx_14[i] < 20.0  # Range
        
        # === HTF TREND BIAS (12h + 1d HMA) ===
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        price_above_hma_1d = close[i] > hma_1d_aligned[i]
        price_below_hma_1d = close[i] < hma_1d_aligned[i]
        
        htf_bullish = price_above_hma_12h and price_above_hma_1d
        htf_bearish = price_below_hma_12h and price_below_hma_1d
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === PRIMARY TREND (4h HMA) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        hma_slope_up = hma_21[i] > hma_21[i-5] if i >= 5 else False
        hma_slope_down = hma_21[i] < hma_21[i-5] if i >= 5 else False
        
        # === RSI PULLBACK SIGNALS ===
        # Long: RSI pulled back to 35-45 in uptrend
        rsi_pullback_long = 35.0 <= rsi_14[i] <= 50.0
        rsi_oversold = rsi_14[i] < 35.0
        rsi_overbought = rsi_14[i] > 65.0
        rsi_extreme_oversold = rsi_14[i] < 30.0
        rsi_extreme_overbought = rsi_14[i] > 70.0
        
        # === VOLUME CONFIRMATION ===
        volume_bullish = taker_ratio[i] > 0.55
        volume_bearish = taker_ratio[i] < 0.45
        
        # === VOL FILTER ===
        vol_ratio = atr_14[i] / (atr_median + 1e-10)
        if vol_ratio > 2.5:
            position_size = BASE_SIZE * 0.5
        elif vol_ratio > 1.5:
            position_size = BASE_SIZE * 0.75
        else:
            position_size = BASE_SIZE
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0
        
        # === LONG ENTRIES ===
        # Condition 1: HTF bullish + HMA bullish + RSI pullback (trend follow)
        if htf_bullish and hma_bullish and rsi_pullback_long:
            signal_strength = 2
            if hma_slope_up:
                signal_strength += 1
            if volume_bullish:
                signal_strength += 1
            if regime_trend:
                signal_strength += 1
            desired_signal = position_size * (0.7 + 0.3 * min(signal_strength, 5) / 5)
        
        # Condition 2: HTF neutral + HMA bullish + RSI oversold (mean reversion)
        elif htf_neutral and hma_bullish and rsi_oversold:
            signal_strength = 1
            if rsi_extreme_oversold:
                signal_strength += 1
            if volume_bullish:
                signal_strength += 1
            desired_signal = position_size * 0.6 * (0.7 + 0.3 * signal_strength / 3)
        
        # Condition 3: Range market + RSI extreme oversold (mean reversion)
        elif regime_range and rsi_extreme_oversold and not htf_bearish:
            signal_strength = 1
            if volume_bullish:
                signal_strength += 1
            desired_signal = position_size * 0.5 * (0.7 + 0.3 * signal_strength / 2)
        
        # === SHORT ENTRIES ===
        # Condition 1: HTF bearish + HMA bearish + RSI pullback (trend follow)
        if desired_signal == 0:
            if htf_bearish and hma_bearish and (50.0 <= rsi_14[i] <= 65.0):
                signal_strength = 2
                if hma_slope_down:
                    signal_strength += 1
                if volume_bearish:
                    signal_strength += 1
                if regime_trend:
                    signal_strength += 1
                desired_signal = -position_size * (0.7 + 0.3 * min(signal_strength, 5) / 5)
            
            # Condition 2: HTF neutral + HMA bearish + RSI overbought (mean reversion)
            elif htf_neutral and hma_bearish and rsi_overbought:
                signal_strength = 1
                if rsi_extreme_overbought:
                    signal_strength += 1
                if volume_bearish:
                    signal_strength += 1
                desired_signal = -position_size * 0.6 * (0.7 + 0.3 * signal_strength / 3)
            
            # Condition 3: Range market + RSI extreme overbought (mean reversion)
            elif regime_range and rsi_extreme_overbought and not htf_bullish:
                signal_strength = 1
                if volume_bearish:
                    signal_strength += 1
                desired_signal = -position_size * 0.5 * (0.7 + 0.3 * signal_strength / 2)
        
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
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === RSI EXTREME EXIT (Take Profit) ===
        if in_position and position_side > 0 and rsi_14[i] > 70.0:
            desired_signal = 0.0
        
        if in_position and position_side < 0 and rsi_14[i] < 30.0:
            desired_signal = 0.0
        
        # === HOLD LOGIC — Maintain position if bias unchanged ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and (hma_bullish or price_above_hma_12h):
                desired_signal = position_size
            elif position_side < 0 and (hma_bearish or price_below_hma_12h):
                desired_signal = -position_size
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal != 0.0:
            if desired_signal > 0:
                if desired_signal >= 0.28:
                    desired_signal = 0.30
                elif desired_signal >= 0.18:
                    desired_signal = 0.25
                else:
                    desired_signal = 0.20
            else:
                if desired_signal <= -0.28:
                    desired_signal = -0.30
                elif desired_signal <= -0.18:
                    desired_signal = -0.25
                else:
                    desired_signal = -0.20
        
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
    
    return signals