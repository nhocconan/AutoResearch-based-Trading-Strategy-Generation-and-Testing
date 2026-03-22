#!/usr/bin/env python3
"""
Experiment #057: 1d Primary + 1w HTF — Dual Regime with Connors RSI

Hypothesis: Daily timeframe with weekly trend filter can capture both mean-reversion
in ranges and trend-following in strong moves. Connors RSI has shown proven results
(Sharpe +0.923 on ETH in research). Choppiness Index provides regime detection.

Strategy Logic:
1. 1w HMA(21) slope determines major trend bias (long/short preference)
2. Choppiness Index(14) detects regime: >61.8 = range (mean revert), <38.2 = trend
3. Connors RSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
   - Long when CRSI < 15 (oversold) + trend bias allows
   - Short when CRSI > 85 (overbought) + trend bias allows
4. In trending regime: use HMA(8/21) crossover for entries
5. ATR(14) stoploss at 2.5x for risk control
6. Position size: 0.28 discrete (conservative for 1d)

Why this should work:
- 1d naturally limits trades to 10-30/year (low fee drag)
- Connors RSI catches reversals better than standard RSI
- Choppiness filter prevents mean-reversion in strong trends
- 1w HTF prevents counter-trend trades in major moves
- Simpler than failed multi-regime strategies but still adaptive

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 15-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_connors_chop_hma_1w_v1"
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

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(100)) / 3
    
    RSI(streak): RSI of consecutive up/down days
    PercentRank: percentage of past 100 days with lower returns
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3) of close
    rsi_close = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    streak_gain = streak_s.where(streak_s > 0, 0.0)
    streak_loss = -streak_s.where(streak_s < 0, 0.0)
    
    avg_streak_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    avg_streak_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = avg_streak_gain / avg_streak_loss.replace(0, np.nan)
    rsi_streak = 100 - (100 / (1 + streak_rs))
    rsi_streak = rsi_streak.fillna(50).values
    
    # Component 3: PercentRank(100) - percentage of past returns lower than current
    returns = close_s.pct_change().fillna(0).values
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = returns[i-pr_period:i]
        current_return = returns[i]
        percent_rank[i] = np.sum(window < current_return) / pr_period * 100
    
    # Combine components
    crsi = (rsi_close + rsi_streak + percent_rank) / 3
    
    # Handle NaN
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    
    CHOP > 61.8 = ranging/choppy market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        # Calculate ATR for each bar in window
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
            chop[i] = 50.0
    
    return chop

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=3):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 2)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    
    # HMA for trend following
    hma_8 = calculate_hma(close, 8)
    hma_21 = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -100
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_8[i]) or np.isnan(hma_21[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # HMA slope > 0 = bullish bias (prefer longs)
        # HMA slope < 0 = bearish bias (prefer shorts)
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.5
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.5
        
        # Price vs 1w HMA for additional confirmation
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        # CHOP > 61.8 = ranging (use mean reversion)
        # CHOP < 38.2 = trending (use trend following)
        # 38.2 < CHOP < 61.8 = neutral (reduce size or skip)
        is_choppy = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        is_neutral = not is_choppy and not is_trending
        
        # === CONNORS RSI SIGNALS ===
        # CRSI < 15 = oversold (long opportunity)
        # CRSI > 85 = overbought (short opportunity)
        crsi_oversold = crsi[i] < 15
        crsi_overbought = crsi[i] > 85
        
        # === HMA CROSSOVER (for trending regime) ===
        hma_bullish_cross = hma_8[i] > hma_21[i] and hma_8[i-1] <= hma_21[i-1]
        hma_bearish_cross = hma_8[i] < hma_21[i] and hma_8[i-1] >= hma_21[i-1]
        hma_aligned_bullish = hma_8[i] > hma_21[i]
        hma_aligned_bearish = hma_8[i] < hma_21[i]
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in neutral regime
        if is_neutral:
            current_size = BASE_SIZE * 0.5
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # MEAN REVERSION ENTRIES (in choppy regime)
        if is_choppy:
            # Long: CRSI oversold + 1w trend not strongly bearish
            if crsi_oversold and not trend_1w_bearish:
                new_signal = current_size
            
            # Short: CRSI overbought + 1w trend not strongly bullish
            if crsi_overbought and not trend_1w_bullish:
                new_signal = -current_size
        
        # TREND FOLLOWING ENTRIES (in trending regime)
        elif is_trending:
            # Long: HMA bullish cross + 1w trend bullish or neutral
            if hma_bullish_cross and (trend_1w_bullish or price_above_1w_hma):
                new_signal = current_size
            
            # Short: HMA bearish cross + 1w trend bearish or neutral
            if hma_bearish_cross and (trend_1w_bearish or price_below_1w_hma):
                new_signal = -current_size
            
            # Pullback entry in established trend
            if hma_aligned_bullish and trend_1w_bullish and crsi[i] < 40:
                new_signal = current_size * 0.7
            elif hma_aligned_bearish and trend_1w_bearish and crsi[i] > 60:
                new_signal = -current_size * 0.7
        
        # NEUTRAL REGIME - wait for clear signals
        elif is_neutral:
            # Only take very strong CRSI signals
            if crsi[i] < 10 and not trend_1w_bearish:
                new_signal = current_size * 0.5
            elif crsi[i] > 90 and not trend_1w_bullish:
                new_signal = -current_size * 0.5
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 200 bars (~200 days on 1d), allow weaker entry
        if bars_since_last_trade > 200 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and hma_aligned_bullish and crsi[i] < 45:
                new_signal = current_size * 0.4
            elif trend_1w_bearish and hma_aligned_bearish and crsi[i] > 55:
                new_signal = -current_size * 0.4
        
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
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1w trend reverses strongly bearish
            if position_side > 0 and trend_1w_bearish and hma_aligned_bearish:
                trend_reversal = True
            # Exit short if 1w trend reverses strongly bullish
            if position_side < 0 and trend_1w_bullish and hma_aligned_bullish:
                trend_reversal = True
        
        # === CRSI EXIT (mean reversion target) ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 75:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 25:
                crsi_exit = True
        
        # Apply stoploss or exits
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