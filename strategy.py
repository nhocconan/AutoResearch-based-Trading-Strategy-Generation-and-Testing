#!/usr/bin/env python3
"""
Experiment #023: 1d Primary + 1w HTF — Volatility Spike Reversion + Connors RSI

Hypothesis: Daily timeframe with weekly trend bias + volatility spike detection should
generate 20-50 trades/year with positive Sharpe on ALL symbols (BTC/ETH/SOL).

Key insight from 21 failed experiments:
- Pure trend following fails on BTC/ETH (especially in 2025 bear market)
- Mean reversion works better in choppy/bear regimes
- Volatility spikes (ATR ratio > 2.0) signal exhaustion → reversion opportunity
- Connors RSI (3-period) catches short-term extremes better than standard RSI(14)

Strategy components:
1. Connors RSI (CRSI): (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - More responsive than RSI(14), catches short-term extremes
   - Entry: CRSI < 15 (long) or CRSI > 85 (short)
2. Volatility Spike Filter: ATR(7)/ATR(30) > 1.8
   - Signals exhaustion after panic/euphoria
   - Only trade mean reversion when vol is elevated
3. 1w HMA: Macro trend bias (trade with weekly direction for higher win rate)
4. Bollinger Band confirmation: Price must touch/extreme BB(20, 2.0)
5. Choppiness Index: Regime detection (range vs trend)
6. ATR stoploss: 2.5*ATR trailing stop

Why this should work:
- 1d primary = fewer trades, less fee drag (targets 25-40/year)
- 1w HTF = strong trend filter, avoids counter-trend trades
- Vol spike + CRSI = high-probability mean reversion setup
- LOOSE CRSI thresholds (15/85 instead of 10/90) = ensures trade generation
- Works in bear markets (2025) via mean reversion, not pure trend

Position size: 0.28 (discrete, within 0.20-0.35 range)
Stoploss: 2.5*ATR trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_crsi_volspike_bb_regime_1w_v1"
timeframe = "1d"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Calculate RSI."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi.values

def calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    Streak RSI: RSI of consecutive up/down days
    PercentRank: Where current close ranks vs last 100 days
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) - very short term momentum
    rsi_3 = calculate_rsi(close, period=rsi_period)
    
    # Streak RSI - consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = calculate_rsi(streak, period=streak_period)
    
    # PercentRank - where current close ranks vs last 100 periods
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        current = close[i]
        rank = np.sum(window < current) / rank_period * 100.0
        percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average (HMA)."""
    close_s = pd.Series(close)
    
    half = int(period / 2)
    sqrt_n = int(np.sqrt(period))
    
    wma_half = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma_full = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2.0 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_n, min_periods=sqrt_n, adjust=False).mean()
    
    return hma.values

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    """
    n = period
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_sum = pd.Series(tr).rolling(window=n, min_periods=n).sum().values
    highest_high = pd.Series(high).rolling(window=n, min_periods=n).max().values
    lowest_low = pd.Series(low).rolling(window=n, min_periods=n).min().values
    
    price_range = highest_high - lowest_low + 1e-10
    chop = 100.0 * np.log10(atr_sum / price_range) / np.log10(n)
    
    return chop

def calculate_percentile_rank(close, period=100):
    """Calculate where current value ranks within rolling window."""
    n = len(close)
    pr = np.zeros(n)
    
    for i in range(period, n):
        window = close[i-period+1:i+1]
        current = close[i]
        pr[i] = np.sum(window < current) / period * 100.0
    
    return pr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA for macro bias
    hma_1w = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    atr_7 = calculate_atr(high, low, close, period=7)
    atr_30 = calculate_atr(high, low, close, period=30)
    
    crsi = calculate_crsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # Volatility spike ratio
    vol_ratio = np.zeros(n)
    for i in range(30, n):
        vol_ratio[i] = atr_7[i] / (atr_30[i] + 1e-10)
    
    # BB %B (where price is within bands)
    bb_pctb = np.zeros(n)
    for i in range(20, n):
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range > 0:
            bb_pctb[i] = (close[i] - bb_lower[i]) / bb_range
        else:
            bb_pctb[i] = 0.5
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    POSITION_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    entry_price = 0.0
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(hma_1w_aligned[i]) or np.isnan(atr_14[i]):
            continue
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]) or np.isnan(bb_upper[i]):
            continue
        if np.isnan(vol_ratio[i]) or atr_14[i] == 0:
            continue
        
        # === 1W MACRO BIAS ===
        price_above_hma_1w = close[i] > hma_1w_aligned[i]
        price_below_hma_1w = close[i] < hma_1w_aligned[i]
        
        # === CHOPPINESS REGIME ===
        chop_value = chop_14[i]
        is_ranging = chop_value > 50.0
        is_trending = chop_value < 45.0
        
        # === VOLATILITY SPIKE ===
        vol_spike = vol_ratio[i] > 1.8  # Elevated volatility
        
        # === CONNORS RSI EXTREMES (LOOSE for trade generation) ===
        crsi_oversold = crsi[i] < 20.0  # Was 15, now 20 for more longs
        crsi_overbought = crsi[i] > 80.0  # Was 85, now 80 for more shorts
        
        # === BOLLINGER BAND EXTREMES ===
        price_below_bb_lower = bb_pctb[i] < 0.05  # Below lower band
        price_above_bb_upper = bb_pctb[i] > 0.95  # Above upper band
        price_near_bb_lower = bb_pctb[i] < 0.15  # Near lower band
        price_near_bb_upper = bb_pctb[i] > 0.85  # Near upper band
        
        # === ADAPTIVE REGIME ENTRY LOGIC ===
        new_signal = 0.0
        
        # --- RANGING REGIME: Mean Reversion (PRIMARY EDGE) ---
        if is_ranging:
            # Long: CRSI oversold + vol spike OR BB extreme
            if crsi_oversold:
                if vol_spike or price_below_bb_lower or price_near_bb_lower:
                    if price_above_hma_1w:  # Weekly bias helps
                        new_signal = POSITION_SIZE
                    elif not price_below_hma_1w:  # Neutral weekly
                        new_signal = POSITION_SIZE * 0.7  # Smaller size
            
            # Short: CRSI overbought + vol spike OR BB extreme
            elif crsi_overbought:
                if vol_spike or price_above_bb_upper or price_near_bb_upper:
                    if price_below_hma_1w:  # Weekly bias helps
                        new_signal = -POSITION_SIZE
                    elif not price_above_hma_1w:  # Neutral weekly
                        new_signal = -POSITION_SIZE * 0.7  # Smaller size
        
        # --- TRENDING REGIME: Pullback Entries ---
        elif is_trending:
            # Long in uptrend: Pullback to BB lower + CRSI recovering
            if price_above_hma_1w:
                if price_near_bb_lower and crsi[i] < 40 and crsi[i] > crsi[i-1]:
                    if vol_spike:  # Vol spike = better entry
                        new_signal = POSITION_SIZE
            
            # Short in downtrend: Rally to BB upper + CRSI turning down
            elif price_below_hma_1w:
                if price_near_bb_upper and crsi[i] > 60 and crsi[i] < crsi[i-1]:
                    if vol_spike:  # Vol spike = better entry
                        new_signal = -POSITION_SIZE
        
        # --- FALLBACK: Simple CRSI extreme without regime filter ---
        if new_signal == 0.0:
            # Long: Very oversold CRSI + BB lower touch
            if crsi[i] < 15 and price_below_bb_lower:
                new_signal = POSITION_SIZE * 0.5  # Smaller size for counter-trend
            
            # Short: Very overbought CRSI + BB upper touch
            elif crsi[i] > 85 and price_above_bb_upper:
                new_signal = -POSITION_SIZE * 0.5  # Smaller size for counter-trend
        
        # === HOLD POSITION LOGIC ===
        if in_position and new_signal == 0.0:
            new_signal = signals[i-1] if i > 0 else 0.0
        
        # === STOPLOSS CHECK (2.5 * ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * atr_14[i]
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            if lowest_since_entry == 0.0:
                lowest_since_entry = close[i]
            else:
                lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * atr_14[i]
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            new_signal = 0.0
        
        # === EXIT ON REGIME CHANGE ===
        # Exit long if weekly trend turns strongly bearish
        if in_position and position_side > 0:
            if price_below_hma_1w and chop_value < 40:  # Strong bear trend
                new_signal = 0.0
        
        # Exit short if weekly trend turns strongly bullish
        if in_position and position_side < 0:
            if price_above_hma_1w and chop_value < 40:  # Strong bull trend
                new_signal = 0.0
        
        # === TAKE PROFIT (reduce at 2R) ===
        if in_position and new_signal != 0.0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit > 2.0 * atr_14[i]:
                    new_signal = POSITION_SIZE * 0.5  # Take half profit
            
            elif position_side < 0:
                profit = entry_price - close[i]
                if profit > 2.0 * atr_14[i]:
                    new_signal = -POSITION_SIZE * 0.5  # Take half profit
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = new_signal
    
    return signals