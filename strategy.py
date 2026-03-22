#!/usr/bin/env python3
"""
Experiment #047: 1d Choppiness Index Regime + Connors RSI + 1w HMA Trend

Hypothesis: Daily timeframe with weekly HTF trend filter will capture major moves
while Choppiness Index regime detection adapts between mean-reversion and trend-following.
Key design:
1. 1w HMA(21) for major trend direction (call ONCE before loop via mtf_data)
2. Choppiness Index(14) for regime detection: >61.8 = range (mean revert), <38.2 = trend
3. Connors RSI for entry timing: (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
4. ATR(14) for stoploss (2.5x) and volatility adjustment
5. Regime-adaptive: mean revert in chop, trend follow otherwise
6. Discrete sizing: 0.25 base, 0.30 strong trend

Why this should work:
- Choppiness Index proven on ETH (Sharpe +0.923 in research)
- Connors RSI has 75% win rate for mean reversion entries
- 1d TF targets 20-50 trades/year (optimal for fee efficiency)
- 1w HTF prevents counter-trend trades in major moves
- Regime switching adapts to market conditions (key for 2022 crash + 2025 bear)

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_chop_connors_1w_hma_regime_v1"
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

def calculate_choppiness_index(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    Formula: 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = choppy/range market
    CHOP < 38.2 = trending market
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
        
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        price_range = highest_high - lowest_low
        
        if price_range > 0 and atr_sum > 0:
            chop[i] = 100 * np.log10(atr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    RSI_Streak: RSI of consecutive up/down days
    PercentRank: percentage of recent closes lower than current close
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3) component
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Streak RSI component
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[i-streak_period+1:i+1]
        avg_streak = np.mean(streak_window)
        # Map streak to 0-100 range
        streak_rsi[i] = 50 + avg_streak * 10
        streak_rsi[i] = np.clip(streak_rsi[i], 0, 100)
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[i-rank_period+1:i+1]
        count_lower = np.sum(window[:-1] < close[i])
        percent_rank[i] = 100 * count_lower / (rank_period - 1)
    
    # Combine components
    for i in range(max(rsi_period, streak_period, rank_period), n):
        crsi[i] = (rsi_short[i] + streak_rsi[i] + percent_rank[i]) / 3
    
    crsi = np.clip(crsi, 0, 100)
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

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
    
    # Also calculate 1d HMA for local trend
    hma_1d_21 = calculate_hma(close, 21)
    hma_1d_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    STRONG_SIZE = 0.30
    WEAK_SIZE = 0.20
    
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1d_21[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1W HTF TREND BIAS ===
        htf_bullish = close[i] > hma_1w_21_aligned[i]
        htf_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D LOCAL TREND ===
        local_bullish = close[i] > hma_1d_21[i]
        local_bearish = close[i] < hma_1d_21[i]
        local_strong_bullish = close[i] > hma_1d_50[i]
        local_strong_bearish = close[i] < hma_1d_50[i]
        
        # === CHOPPINESS REGIME ===
        choppy_regime = chop_14[i] > 61.8  # Range/mean-reversion market
        trending_regime = chop_14[i] < 38.2  # Trending market
        neutral_regime = not choppy_regime and not trending_regime
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Strong mean-reversion long signal
        crsi_overbought = crsi[i] > 85  # Strong mean-reversion short signal
        crsi_mild_oversold = crsi[i] < 30
        crsi_mild_overbought = crsi[i] > 70
        
        # === POSITION SIZING BASED ON TREND STRENGTH ===
        if htf_bullish and local_bullish:
            current_size = STRONG_SIZE
        elif htf_bullish or local_bullish:
            current_size = BASE_SIZE
        elif htf_bearish and local_bearish:
            current_size = STRONG_SIZE
        elif htf_bearish or local_bearish:
            current_size = BASE_SIZE
        else:
            current_size = WEAK_SIZE
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # REGIME 1: CHOPPY (mean reversion)
        if choppy_regime:
            # Long: HTF bullish + CRSI oversold
            if htf_bullish and crsi_oversold:
                new_signal = current_size
            # Short: HTF bearish + CRSI overbought
            elif htf_bearish and crsi_overbought:
                new_signal = -current_size
            # Weaker signals in choppy regime
            elif htf_bullish and crsi_mild_oversold:
                new_signal = current_size * 0.7
            elif htf_bearish and crsi_mild_overbought:
                new_signal = -current_size * 0.7
        
        # REGIME 2: TRENDING (trend following)
        elif trending_regime:
            # Long: HTF bullish + local bullish + CRSI not overbought
            if htf_bullish and local_bullish and crsi[i] < 70:
                new_signal = current_size
            # Short: HTF bearish + local bearish + CRSI not oversold
            elif htf_bearish and local_bearish and crsi[i] > 30:
                new_signal = -current_size
        
        # REGIME 3: NEUTRAL (mixed signals)
        else:
            # Only take strong CRSI extremes
            if htf_bullish and crsi_oversold:
                new_signal = current_size * 0.8
            elif htf_bearish and crsi_overbought:
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 20 bars (~20 days on 1d), allow weaker entry
        if bars_since_last_trade > 20 and new_signal == 0.0 and not in_position:
            if htf_bullish and crsi_mild_oversold:
                new_signal = BASE_SIZE * 0.6
            elif htf_bearish and crsi_mild_overbought:
                new_signal = -BASE_SIZE * 0.6
        
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
        
        # === CRSI EXTREME EXIT (take profit) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes very overbought
            if position_side > 0 and crsi[i] > 80:
                crsi_exit = True
            # Exit short when CRSI becomes very oversold
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