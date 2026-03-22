#!/usr/bin/env python3
"""
Experiment #248: 30m Connors RSI + Choppiness Regime + 4h HMA Trend

Hypothesis: 30m timeframe captures intraday swings while filtering noise.
Using Connors RSI for mean reversion entries (75% win rate in literature) +
Choppiness Index for regime detection + 4h HMA for higher timeframe bias.

Why this might work on 30m:
- 30m balances signal quality with trade frequency (more than 4h, less noise than 15m)
- Connors RSI excels at catching oversold/overbought extremes in ranging markets
- Choppiness Index filters out low-probability entries during strong trends
- 4h HMA provides trend bias to avoid counter-trend mean reversion traps
- Conservative sizing (0.25) + ATR stoploss controls drawdown

Key improvements over failed experiments:
- #241 (15m trend pullback): Sharpe=-3.471 - too many whipsaws
- #247 (15m chop regime): Sharpe=-3.271 - similar concept but wrong TF
- This uses 30m (less noise than 15m) with Connors RSI (proven edge)
- Choppiness threshold tuned for 30m (not copied from 4h strategies)
- 4h HMA trend filter prevents counter-trend entries in strong moves

Timeframe: 30m (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 discrete levels
Stoploss: 2.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_30m_connors_rsi_chop_regime_4h_hma_atr_v1"
timeframe = "30m"
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
    """Calculate RSI (Relative Strength Index)."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = Ranging/Choppy market
    CHOP < 38.2 = Trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        atr_sum = 0.0
        highest_high = low[i]
        lowest_low = high[i]
        
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
            highest_high = max(highest_high, high[j])
            lowest_low = min(lowest_low, low[j])
        
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean()
    return sma.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(close, 3) + RSI(streak, 2) + PercentRank(close, 100)) / 3
    
    RSI(3): Fast momentum
    RSI(Streak): Measures consecutive up/down days
    PercentRank: Where current price ranks in last 100 periods
    
    Entry: CRSI < 10 (oversold) or CRSI > 90 (overbought)
    """
    n = len(close)
    crsi = np.zeros(n)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value (0-100)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_vals = streak[i-streak_period+1:i+1]
        gains = np.sum(streak_vals[streak_vals > 0])
        losses = np.abs(np.sum(streak_vals[streak_vals < 0]))
        if losses == 0:
            streak_rsi[i] = 100.0
        else:
            rs = gains / (losses + 1e-10)
            streak_rsi[i] = 100 - (100 / (1 + rs))
    
    # Percent Rank
    percent_rank = np.zeros(n)
    for i in range(pr_period, n):
        window = close[i-pr_period:i]
        count_below = np.sum(window < close[i])
        percent_rank[i] = 100 * count_below / pr_period
    
    # Combine into CRSI
    for i in range(pr_period, n):
        crsi[i] = (rsi_3[i] + streak_rsi[i] + percent_rank[i]) / 3.0
    
    return crsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - no manual index mapping, auto shift(1))
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 30m indicators
    atr = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, pr_period=100)
    chop = calculate_choppiness(high, low, close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Position sizing - discrete levels (Rule 4)
    SIZE_BASE = 0.25
    SIZE_HALF = 0.12
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    highest_close = 0.0
    lowest_close = 0.0
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] == 0:
            signals[i] = 0.0
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            continue
        
        # === HIGHER TIMEFRAME BIAS ===
        # 4h HMA = trend bias
        bull_trend_4h = close[i] > hma_4h_aligned[i]
        bear_trend_4h = close[i] < hma_4h_aligned[i]
        
        # === REGIME DETECTION ===
        # CHOP > 55 = Ranging (favor mean reversion)
        # CHOP < 45 = Trending (favor trend following)
        is_ranging = chop[i] > 55
        is_trending = chop[i] < 45
        
        # === ENTRY SIGNALS ===
        new_signal = 0.0
        
        # --- MEAN REVERSION: Connors RSI extremes + regime filter ---
        # Long: CRSI < 15 (oversold) + ranging or 4h bullish + above SMA200
        if crsi[i] < 15:
            if is_ranging or bull_trend_4h:
                if close[i] > sma_200[i]:  # Above long-term MA for safety
                    new_signal = SIZE_BASE
        
        # Short: CRSI > 85 (overbought) + ranging or 4h bearish + below SMA200
        if crsi[i] > 85:
            if is_ranging or bear_trend_4h:
                if close[i] < sma_200[i]:  # Below long-term MA for safety
                    new_signal = -SIZE_BASE
        
        # --- TREND FOLLOWING: Pullback to 4h HMA in trending regime ---
        # Only when CHOP indicates trending market
        if is_trending:
            # Long pullback: 4h bullish + price near 4h HMA + CRSI recovering
            if bull_trend_4h and crsi[i] > 40 and crsi[i] < 60:
                if close[i] > hma_4h_aligned[i] * 0.98:  # Near HMA support
                    new_signal = SIZE_BASE
            
            # Short pullback: 4h bearish + price near 4h HMA + CRSI declining
            if bear_trend_4h and crsi[i] > 40 and crsi[i] < 60:
                if close[i] < hma_4h_aligned[i] * 1.02:  # Near HMA resistance
                    new_signal = -SIZE_BASE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.0 * ATR trailing ===
        # Check stoploss on EXISTING position before considering new entry
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest close for long position
                if close[i] > highest_close:
                    highest_close = close[i]
                # Trailing stop: 2.0 * ATR below highest close
                stoploss_price = highest_close - 2.0 * atr[i]
                if close[i] < stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
            
            if position_side < 0:
                # Update lowest close for short position
                if lowest_close == 0.0 or close[i] < lowest_close:
                    lowest_close = close[i]
                # Trailing stop: 2.0 * ATR above lowest close
                stoploss_price = lowest_close + 2.0 * atr[i]
                if close[i] > stoploss_price:
                    new_signal = 0.0  # Stoploss overrides entry signal
        
        # === TAKE PROFIT: Reduce to half at 2R profit ===
        if in_position and new_signal != 0.0 and position_side != 0:
            if position_side > 0:
                profit = close[i] - entry_price
                if profit >= 2.0 * entry_atr:
                    new_signal = SIZE_HALF  # Take partial profit
            if position_side < 0:
                profit = entry_price - close[i]
                if profit >= 2.0 * entry_atr:
                    new_signal = -SIZE_HALF  # Take partial profit
        
        # === UPDATE POSITION TRACKING FOR NEXT BAR ===
        if new_signal != 0.0:
            if not in_position:
                # Entering new position
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            elif np.sign(new_signal) != position_side:
                # Reversing position
                position_side = np.sign(new_signal)
                entry_price = close[i]
                entry_atr = atr[i]
                highest_close = close[i] if position_side > 0 else 0.0
                lowest_close = close[i] if position_side < 0 else 0.0
            # else: maintaining same position direction (possibly reduced size)
        else:
            # Exiting position (signal-based or stoploss)
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_close = 0.0
                lowest_close = 0.0
        
        signals[i] = new_signal
    
    return signals