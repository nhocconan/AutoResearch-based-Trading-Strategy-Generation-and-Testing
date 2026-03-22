#!/usr/bin/env python3
"""
Experiment #032: 12h Dual Regime + Connors RSI + 1d/1w Bias

Hypothesis: 12h timeframe provides optimal trade frequency (20-50/year) with less
noise than lower TFs. Dual regime approach adapts to market conditions:
1. CHOP > 61.8 = range → Connors RSI mean reversion (oversold/overbought extremes)
2. CHOP < 38.2 = trend → HMA crossover + RSI momentum confirmation
3. 1d HMA for intermediate trend bias
4. 1w HMA for major trend bias (prevents counter-trend in major moves)
5. ATR(14) trailing stoploss at 2.5x
6. Discrete position sizing (0.25-0.30) to minimize fee churn

Why this should work:
- 12h TF = natural 20-50 trades/year (avoids fee drag from lower TFs)
- Connors RSI has 75% win rate in literature for mean reversion
- Choppiness Index proven regime filter (ETH Sharpe +0.923 in history)
- 1d/1w dual bias prevents major counter-trend losses
- Looser entry thresholds ensure minimum trade count (avoiding 0-trade failures)

Timeframe: 12h (REQUIRED for this experiment)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_dual_regime_connors_1d1w_bias_v1"
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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long entry: CRSI < 10 (oversold)
    Short entry: CRSI > 90 (overbought)
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Streak RSI - count consecutive up/down days
    streak = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Convert streak to RSI-like value
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        streak_window = streak[max(0, i-streak_period+1):i+1]
        pos_count = np.sum(streak_window > 0)
        if streak_period > 0:
            streak_rsi[i] = 100 * pos_count / streak_period if np.any(streak_window != 0) else 50
        else:
            streak_rsi[i] = 50
    
    # Percent Rank - where current price ranks in last 100 bars
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        current = close[i]
        rank = np.sum(window < current) / len(window) * 100
        percent_rank[i] = rank
    
    # Combine into CRSI
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = range/choppy market
    CHOP < 38.2 = trending market
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # ATR over period
    atr = calculate_atr(high, low, close, period)
    atr_s = pd.Series(atr)
    atr_sum = atr_s.rolling(window=period, min_periods=period).sum()
    
    # Choppiness formula
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(period)
    chop = chop.fillna(50).values
    
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

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high, lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max().values
    lower = low_s.rolling(window=period, min_periods=period).min().values
    
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1D indicators
    hma_1d_50 = calculate_hma(df_1d['close'].values, 50)
    
    # Calculate 1W indicators
    hma_1w_20 = calculate_hma(df_1w['close'].values, 20)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1d_50_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_50)
    hma_1w_20_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_20)
    
    # Calculate 12h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, 14)
    hma_12h_21 = calculate_hma(close, 21)
    hma_12h_50 = calculate_hma(close, 50)
    hma_12h_200 = calculate_hma(close, 200)
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.28
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_50_aligned[i]) or np.isnan(hma_1w_20_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(hma_12h_21[i]) or np.isnan(hma_12h_50[i]):
            continue
        
        # === 1W MAJOR TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_20_aligned[i]
        weekly_bearish = close[i] < hma_1w_20_aligned[i]
        
        # === 1D INTERMEDIATE TREND BIAS ===
        daily_bullish = close[i] > hma_1d_50_aligned[i]
        daily_bearish = close[i] < hma_1d_50_aligned[i]
        
        # === 12H HMA TREND ===
        hma_bullish = hma_12h_21[i] > hma_12h_50[i]
        hma_bearish = hma_12h_21[i] < hma_12h_50[i]
        
        # === CHOPPINNESS REGIME ===
        choppy_market = chop_14[i] > 61.8
        trending_market = chop_14[i] < 38.2
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        if weekly_bullish or daily_bullish:  # At least one HTF bullish
            if choppy_market:
                # Mean reversion: Connors RSI oversold in range
                if crsi[i] < 15 and close[i] > hma_12h_200[i]:
                    new_signal = current_size
                # Alternative: RSI oversold
                elif rsi_14[i] < 30 and close[i] > hma_12h_200[i]:
                    new_signal = current_size * 0.8
            elif trending_market:
                # Trend follow: HMA crossover + RSI confirmation
                if hma_bullish and 40 <= rsi_14[i] <= 70:
                    new_signal = current_size
                # Donchian breakout in uptrend
                elif close[i] > donchian_upper[i] * 0.995 and hma_bullish:
                    new_signal = current_size
        
        # SHORT ENTRIES
        elif weekly_bearish or daily_bearish:  # At least one HTF bearish
            if choppy_market:
                # Mean reversion: Connors RSI overbought in range
                if crsi[i] > 85 and close[i] < hma_12h_200[i]:
                    new_signal = -current_size
                # Alternative: RSI overbought
                elif rsi_14[i] > 70 and close[i] < hma_12h_200[i]:
                    new_signal = -current_size * 0.8
            elif trending_market:
                # Trend follow: HMA crossover + RSI confirmation
                if hma_bearish and 30 <= rsi_14[i] <= 60:
                    new_signal = -current_size
                # Donchian breakdown in downtrend
                elif close[i] < donchian_lower[i] * 1.005 and hma_bearish:
                    new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 50 bars (~25 days on 12h), force entry with weaker signal
        if bars_since_last_trade > 50 and new_signal == 0.0 and not in_position:
            if (weekly_bullish or daily_bullish) and rsi_14[i] > 35:
                new_signal = current_size * 0.5
            elif (weekly_bearish or daily_bearish) and rsi_14[i] < 65:
                new_signal = -current_size * 0.5
        
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
            if position_side > 0 and hma_bearish and (weekly_bearish and daily_bearish):
                trend_reversal = True
            if position_side < 0 and hma_bullish and (weekly_bullish and daily_bullish):
                trend_reversal = True
        
        # Apply stoploss or trend reversal
        if stoploss_triggered or trend_reversal:
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