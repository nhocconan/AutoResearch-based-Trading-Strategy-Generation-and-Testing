#!/usr/bin/env python3
"""
Experiment #226: 12h Primary + 1d HTF — Choppiness Regime + Connors RSI + HMA Trend

Hypothesis: Based on research showing Connors RSI + Choppiness Index achieved 
ETH Sharpe +0.923, this strategy combines:

1. CHOPPINESS INDEX (14): Regime detection
   - CHOP > 61.8 = range market → mean reversion logic
   - CHOP < 38.2 = trending market → trend following logic
   
2. CONNORS RSI (CRSI): Superior to standard RSI for entries
   - CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long: CRSI < 15 (oversold) + trend filter
   - Short: CRSI > 85 (overbought) + trend filter
   
3. 1d HTF HMA(21): Major trend alignment
   - Only long when price > 1d HMA
   - Only short when price < 1d HMA
   
4. ATR(14) Trailing Stop: 2.5 * ATR protection

Why 12h timeframe:
- Natural balance between trade frequency (20-50/year) and signal quality
- Less noise than 4h, more trades than 1d
- Works well with daily HTF confirmation

Position sizing: 0.28 discrete (max 0.35)
Stoploss: 2.5 * ATR(14) trailing
Target: 25-45 trades/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_chop_connors_hma_1d_v2"
timeframe = "12h"
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
    """Calculate RSI using standard Wilder's method."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR(1), period) / (Highest High - Lowest Low)) * 100
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    # Calculate true range for each bar (ATR(1))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Sum of ATR(1) over period
    tr_series = pd.Series(tr)
    sum_atr = tr_series.rolling(window=period, min_periods=period).sum().values
    
    # Highest high and lowest low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Calculate CHOP
    price_range = highest_high - lowest_low
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(sum_atr / price_range) / np.log10(period)
    
    # Handle edge cases
    chop = np.where(price_range == 0, 50.0, chop)
    chop = np.where(np.isnan(chop), 50.0, chop)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: Where current return ranks vs last 100 bars
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    # Streak: count consecutive up or down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    # Calculate RSI of streak values
    streak_rsi = calculate_rsi(streak, streak_period)
    
    # Component 3: Percent Rank of returns over last 100 bars
    returns = close_s.pct_change().values
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = returns[i-rank_period:i]
        current_return = returns[i]
        if len(window) > 0 and not np.isnan(current_return):
            rank = np.sum(window < current_return) / len(window) * 100
            percent_rank[i] = rank
        else:
            percent_rank[i] = 50.0
    
    # Combine into CRSI
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d HTF indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # 12h HMA for local trend
    hma_12h_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_12h_21[i]):
            continue
        
        # === REGIME DETECTION (Choppiness Index) ===
        is_choppy = chop_14[i] > 55.0  # Range market
        is_trending = chop_14[i] < 45.0  # Trend market
        # Neutral zone between 45-55
        
        # === HTF TREND BIAS (1d) ===
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === LOCAL TREND (12h HMA) ===
        price_above_12h_hma = close[i] > hma_12h_21[i]
        price_below_12h_hma = close[i] < hma_12h_21[i]
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20  # Strong oversold
        crsi_overbought = crsi[i] > 80  # Strong overbought
        crsi_extreme_oversold = crsi[i] < 12
        crsi_extreme_overbought = crsi[i] > 88
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Different logic per regime
        long_signal = False
        long_strength = 0
        
        if is_choppy:
            # MEAN REVERSION in choppy market
            # Long when CRSI extremely oversold + price near support
            if crsi_extreme_oversold and price_above_1d_hma:
                long_signal = True
                long_strength = 4
            elif crsi_oversold and price_above_1d_hma and price_above_12h_hma:
                long_signal = True
                long_strength = 3
            elif crsi_oversold and price_above_1d_hma and bars_since_last_trade > 40:
                long_signal = True
                long_strength = 2
        elif is_trending:
            # TREND FOLLOWING in trending market
            # Long when CRSI pulls back but trend intact
            if crsi_oversold and price_above_1d_hma and price_above_12h_hma:
                long_signal = True
                long_strength = 4
            elif crsi[i] < 35 and price_above_1d_hma and price_above_12h_hma:
                long_signal = True
                long_strength = 3
            elif price_above_1d_hma and price_above_12h_hma and crsi[i] < 45:
                long_signal = True
                long_strength = 2
        else:
            # NEUTRAL regime - require stronger signals
            if crsi_extreme_oversold and price_above_1d_hma:
                long_signal = True
                long_strength = 3
            elif crsi_oversold and price_above_1d_hma and price_above_12h_hma:
                long_signal = True
                long_strength = 2
        
        if long_signal:
            if long_strength >= 3:
                new_signal = current_size
            elif long_strength == 2 and bars_since_last_trade > 35:
                new_signal = current_size * 0.6
            elif long_strength >= 1 and bars_since_last_trade > 50:
                new_signal = current_size * 0.4
        
        # SHORT ENTRIES
        short_signal = False
        short_strength = 0
        
        if is_choppy:
            # MEAN REVERSION in choppy market
            if crsi_extreme_overbought and price_below_1d_hma:
                short_signal = True
                short_strength = 4
            elif crsi_overbought and price_below_1d_hma and price_below_12h_hma:
                short_signal = True
                short_strength = 3
            elif crsi_overbought and price_below_1d_hma and bars_since_last_trade > 40:
                short_signal = True
                short_strength = 2
        elif is_trending:
            # TREND FOLLOWING in trending market
            if crsi_overbought and price_below_1d_hma and price_below_12h_hma:
                short_signal = True
                short_strength = 4
            elif crsi[i] > 65 and price_below_1d_hma and price_below_12h_hma:
                short_signal = True
                short_strength = 3
            elif price_below_1d_hma and price_below_12h_hma and crsi[i] > 55:
                short_signal = True
                short_strength = 2
        else:
            # NEUTRAL regime - require stronger signals
            if crsi_extreme_overbought and price_below_1d_hma:
                short_signal = True
                short_strength = 3
            elif crsi_overbought and price_below_1d_hma and price_below_12h_hma:
                short_signal = True
                short_strength = 2
        
        if short_signal:
            if short_strength >= 3:
                new_signal = -current_size
            elif short_strength == 2 and bars_since_last_trade > 35:
                new_signal = -current_size * 0.6
            elif short_strength >= 1 and bars_since_last_trade > 50:
                new_signal = -current_size * 0.4
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 70 bars (~35 days on 12h)
        if bars_since_last_trade > 70 and new_signal == 0.0 and not in_position:
            if price_above_1d_hma and crsi[i] < 40:
                new_signal = current_size * 0.35
            elif price_below_1d_hma and crsi[i] > 60:
                new_signal = -current_size * 0.35
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === HTF TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Long position but daily turns bearish
            if position_side > 0 and price_below_1d_hma and chop_14[i] < 40:
                trend_reversal = True
            # Short position but daily turns bullish
            if position_side < 0 and price_above_1d_hma and chop_14[i] < 40:
                trend_reversal = True
        
        if stoploss_triggered or trend_reversal:
            new_signal = 0.0
        
        # === CRSI REVERSAL EXIT ===
        # Exit long when CRSI becomes overbought
        if in_position and position_side > 0 and crsi[i] > 75:
            new_signal = 0.0
        # Exit short when CRSI becomes oversold
        if in_position and position_side < 0 and crsi[i] < 25:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals