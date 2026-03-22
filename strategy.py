#!/usr/bin/env python3
"""
Experiment #044: 4h Choppiness Regime + 12h HMA Trend + Connors RSI Entries

Hypothesis: 4h primary timeframe with 12h trend filter and Connors RSI entries
will capture both trending and ranging markets with adaptive logic.

Key design:
1. 12h HMA(21) for major trend bias (call ONCE via mtf_data)
2. Choppiness Index(14) for regime detection (>55 = range, <45 = trend)
3. Connors RSI (RSI2 + RSI_Streak + PercentRank) for entry timing
4. ATR(14) for stoploss (2.5x)
5. Discrete sizing: 0.25 base, 0.30 strong trend

Why this should work:
- 4h TF naturally limits trades to 30-60/year (fee efficient)
- 12h HTF filter prevents counter-trend trades in strong trends
- Choppiness adapts between mean-revert and trend-follow modes
- Connors RSI more responsive than standard RSI for entry timing
- Wide CRSI thresholds (10-30 long, 70-90 short) ensure trades trigger
- Frequency safeguard after 40 bars without trades

Timeframe: 4h (REQUIRED)
HTF: 12h via mtf_data helper
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_chop_regime_12h_hma_crsi_v1"
timeframe = "4h"
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
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    for i in range(period, n):
        atr_sum = np.sum(tr[i-period+1:i+1])
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50  # neutral
    
    return chop

def calculate_connors_rsi(close, rsi_period=2, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 2) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(close, 2): 2-period RSI on close
    RSI(streak, 2): 2-period RSI on streak (consecutive up/down days)
    PercentRank(100): percentile rank of today's return over last 100 days
    
    CRSI < 10 = extremely oversold (long opportunity)
    CRSI > 90 = extremely overbought (short opportunity)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(2) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Calculate streak
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # RSI(2) on streak (convert to positive for RSI calculation)
    streak_positive = streak + np.abs(streak.min()) + 1
    rsi_streak = calculate_rsi(streak_positive, streak_period)
    
    # PercentRank(100)
    returns = np.diff(close) / np.roll(close, 1)
    returns[0] = 0
    for i in range(rank_period, n):
        window = returns[i-rank_period+1:i+1]
        current_return = returns[i]
        rank = np.sum(window < current_return)
        percent_rank = rank / rank_period
        crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank * 100) / 3
    
    return np.clip(crsi, 0, 100)

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h HMA trend
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    
    # Align HTF to LTF (auto shift(1) for completed bars only)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=2, streak_period=2, rank_period=100)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    
    # Track position state for stoploss
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
        
        if np.isnan(hma_12h_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === HTF TREND BIAS (12h) ===
        htf_bullish = close[i] > hma_12h_aligned[i]
        htf_bearish = close[i] < hma_12h_aligned[i]
        
        # === REGIME DETECTION (Choppiness) ===
        # CHOP > 55 = ranging (mean revert)
        # CHOP < 45 = trending (trend follow)
        # 45 - 55 = neutral (use trend bias)
        is_choppy = chop_14[i] > 55
        is_trending = chop_14[i] < 45
        
        # === ENTRY LOGIC - REGIME ADAPTIVE (wide thresholds for trade gen) ===
        new_signal = 0.0
        
        if is_trending and htf_bullish:
            # Trend follow long: CRSI pullback in uptrend (wide range)
            if 15 <= crsi[i] <= 50:
                new_signal = STRONG_SIZE
        
        elif is_trending and htf_bearish:
            # Trend follow short: CRSI rally in downtrend (wide range)
            if 50 <= crsi[i] <= 85:
                new_signal = -STRONG_SIZE
        
        elif is_choppy:
            # Mean reversion in range (wider thresholds for more trades)
            if crsi[i] < 35:
                new_signal = BASE_SIZE  # long at oversold
            elif crsi[i] > 65:
                new_signal = -BASE_SIZE  # short at overbought
        
        else:
            # Neutral regime: use HTF bias with moderate CRSI
            if htf_bullish and crsi[i] < 45:
                new_signal = BASE_SIZE
            elif htf_bearish and crsi[i] > 55:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 40 bars (~40 * 4h = 160 hours = ~7 days), force entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 40 and new_signal == 0.0 and not in_position:
            if htf_bullish:
                new_signal = BASE_SIZE * 0.8
            elif htf_bearish:
                new_signal = -BASE_SIZE * 0.8
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR ===
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if HTF trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if HTF trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes very overbought
            if position_side > 0 and crsi[i] > 85:
                crsi_exit = True
            # Exit short when CRSI becomes very oversold
            if position_side < 0 and crsi[i] < 15:
                crsi_exit = True
        
        # Apply stoploss or reversals
        if stoploss_triggered or trend_reversal or crsi_exit:
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