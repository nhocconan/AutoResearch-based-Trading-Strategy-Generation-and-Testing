#!/usr/bin/env python3
"""
Experiment #003: 1d Choppiness Index Regime + Connors RSI + 1w HMA Trend Filter

Hypothesis: Daily timeframe with weekly trend filter + regime-adaptive entries will work
across bull/bear/range markets. Key design:
1. 1w HMA(21) for major trend direction (call ONCE before loop via mtf_data)
2. Choppiness Index(14) for regime detection: >61.8=range, <38.2=trend
3. Connors RSI for entries: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. Regime-adaptive logic: mean revert in chop, trend follow otherwise
5. ATR(14) for stoploss (2.5x) and position sizing
6. Discrete sizing: 0.25 base, 0.30 strong confluence

Why this should work:
- Choppiness Index filters out whipsaw in range markets (major failure mode 2022)
- Connors RSI has 75% win rate for mean reversion entries
- 1w HMA prevents counter-trend trades in major moves
- 1d TF targets 20-50 trades/year (optimal for fee efficiency)
- Proven on ETH with Sharpe +0.923 in similar configuration

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_connors_1w_hma_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    > 61.8 = choppy/range, < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in the window
        atr_sum = 0.0
        for j in range(i - period + 1, i + 1):
            tr1 = high[j] - low[j]
            tr2 = abs(high[j] - close[j-1]) if j > 0 else tr1
            tr3 = abs(low[j] - close[j-1]) if j > 0 else tr1
            tr = max(tr1, tr2, tr3)
            atr_sum += tr
        
        hh = np.max(high[i-period+1:i+1])
        ll = np.min(low[i-period+1:i+1])
        range_hl = hh - ll
        
        if range_hl > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(Streak, 2) + PercentRank(close, 100)) / 3
    
    Streak: consecutive up/down days (positive for up, negative for down)
    PercentRank: percentage of closes in last 100 days that are <= current close
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) on close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Streak calculation
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # RSI(2) on streak (need to handle negative values)
    streak_shifted = streak - np.min(streak) + 1  # make all positive
    rsi_streak = calculate_rsi(streak_shifted, streak_period)
    
    # PercentRank(100)
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_below = np.sum(window <= close[i])
        percent_rank[i] = 100 * count_below / rank_period
    
    # Combine
    for i in range(rank_period, n):
        crsi[i] = (rsi_close[i] + rsi_streak[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA trend
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness_index(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # Also calculate 1d HMA for additional confirmation
    hma_1d_21 = calculate_hma(close, 21)
    
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
    
    for i in range(150, n):  # Need 150 bars for CRSI rank_period=100 + buffers
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1W HTF TREND BIAS ===
        # Simple: price above 1w HMA = bullish bias, below = bearish
        htf_bullish = close[i] > hma_1w_21_aligned[i]
        htf_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        local_bullish = close[i] > hma_1d_21[i]
        local_bearish = close[i] < hma_1d_21[i]
        
        # === CHOPPINESS REGIME ===
        regime_chop = chop_14[i] > 61.8  # Range market
        regime_trend = chop_14[i] < 38.2  # Trending market
        regime_neutral = not regime_chop and not regime_trend
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Strong mean reversion long signal
        crsi_overbought = crsi[i] > 85  # Strong mean reversion short signal
        crsi_moderate_oversold = crsi[i] < 25
        crsi_moderate_overbought = crsi[i] > 75
        
        # === POSITION SIZING BASED ON CONFLUENCE ===
        current_size = BASE_SIZE
        if htf_bullish and local_bullish:
            current_size = STRONG_SIZE
        elif htf_bearish and local_bearish:
            current_size = STRONG_SIZE
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if regime_chop:
            # Range market: mean reversion at extremes
            if crsi_oversold and htf_bullish:
                new_signal = current_size
            elif crsi_moderate_oversold and htf_bullish and local_bullish:
                new_signal = current_size * 0.8
        elif regime_trend:
            # Trending market: pullback entries in direction of trend
            if htf_bullish and crsi_moderate_oversold and local_bullish:
                new_signal = current_size
        else:
            # Neutral regime: require stronger confluence
            if crsi_oversold and htf_bullish and local_bullish:
                new_signal = current_size
        
        # SHORT ENTRIES
        if regime_chop:
            # Range market: mean reversion at extremes
            if crsi_overbought and htf_bearish:
                new_signal = -current_size
            elif crsi_moderate_overbought and htf_bearish and local_bearish:
                new_signal = -current_size * 0.8
        elif regime_trend:
            # Trending market: pullback entries in direction of trend
            if htf_bearish and crsi_moderate_overbought and local_bearish:
                new_signal = -current_size
        else:
            # Neutral regime: require stronger confluence
            if crsi_overbought and htf_bearish and local_bearish:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 45 bars (~45 days on 1d), allow weaker entry
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if crsi_moderate_oversold and htf_bullish:
                new_signal = current_size * 0.7
            elif crsi_moderate_overbought and htf_bearish:
                new_signal = -current_size * 0.7
        
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
            # Exit long if 1w trend turns bearish
            if position_side > 0 and htf_bearish:
                trend_reversal = True
            # Exit short if 1w trend turns bullish
            if position_side < 0 and htf_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes very overbought (mean reversion complete)
            if position_side > 0 and crsi[i] > 80:
                crsi_exit = True
            # Exit short when CRSI becomes very oversold (mean reversion complete)
            if position_side < 0 and crsi[i] < 20:
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