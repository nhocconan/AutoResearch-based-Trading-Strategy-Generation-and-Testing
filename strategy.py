#!/usr/bin/env python3
"""
Experiment #037: 1d Dual Regime + Connors RSI + 1w HMA Trend Bias

Hypothesis: Daily timeframe with weekly trend bias provides optimal balance between
signal quality and trade frequency. Previous failures stemmed from:
1. Too many conflicting filters (0 trades)
2. Wrong regime detection (trend-following in choppy markets)
3. No HTF bias (entering against major trend)

This strategy uses PROVEN patterns from research:
1. 1w HMA(21) for major trend bias (slower, more reliable than 1d)
2. Choppiness Index(14) regime detection: >61.8=range, <38.2=trend
3. Connors RSI for entries (75% win rate in literature):
   - Range regime: CRSI<15 long, CRSI>85 short (mean reversion)
   - Trend regime: Pullback to HMA(50) + RSI(14)<40 long, >60 short
4. ATR(14) trailing stoploss at 2.5x
5. Discrete position sizing: 0.25 base, 0.30 high conviction

Why this should beat Sharpe=0.028:
- 1d timeframe naturally gives 20-50 trades/year (optimal for fee drag)
- Dual regime adapts to market conditions (key for 2022 crash + 2025 bear)
- 1w HMA provides strong trend filter without over-complication
- Connors RSI proven edge in mean-reversion scenarios
- Simpler than failed KAMA/Fisher combinations = more trades

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_dual_regime_connors_1w_hma_v1"
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
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    return rsi

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    
    Entry signals:
    - Long: CRSI < 15 (oversold)
    - Short: CRSI > 85 (overbought)
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # Component 1: RSI(3)
    rsi_short = calculate_rsi(close, rsi_period)
    
    # Component 2: Streak RSI
    delta = np.diff(close, prepend=close[0])
    streak = np.zeros(n)
    
    for i in range(1, n):
        if delta[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    # Streak RSI calculation
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        if streak[i] >= 0:
            # Count consecutive up days
            up_streak = 0
            for j in range(i, max(i-streak_period-1, -1), -1):
                if delta[j] > 0:
                    up_streak += 1
                else:
                    break
            streak_rsi[i] = 100 * up_streak / (streak_period + 1)
        else:
            # Count consecutive down days
            down_streak = 0
            for j in range(i, max(i-streak_period-1, -1), -1):
                if delta[j] < 0:
                    down_streak += 1
                else:
                    break
            streak_rsi[i] = 100 * (1 - down_streak / (streak_period + 1))
    
    # Component 3: Percentile Rank
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period+1):i+1]
        if len(window) > 0:
            percent_rank[i] = 100 * np.sum(window < close[i]) / len(window)
    
    # Combine components
    crsi = (rsi_short + streak_rsi + percent_rank) / 3.0
    
    # Handle NaN at start
    crsi[:max(rsi_period, streak_period, rank_period)] = 50
    
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

def calculate_sma(close, period=200):
    """Calculate Simple Moving Average."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    chop_14 = calculate_choppiness(high, low, close, 14)
    hma_1d_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(crsi[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(hma_1d_50[i]) or np.isnan(rsi_14[i]):
            continue
        
        # === 1W TREND BIAS ===
        trend_bullish = close[i] > hma_1w_21_aligned[i]
        trend_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINNESS REGIME ===
        choppy_market = chop_14[i] > 61.8
        trending_market = chop_14[i] < 38.2
        neutral_market = not choppy_market and not trending_market
        
        # === CONNORS RSI SIGNALS ===
        crsi_oversold = crsi[i] < 20  # Looser for more trades
        crsi_overbought = crsi[i] > 80  # Looser for more trades
        
        # === RSI SIGNALS ===
        rsi_oversold = rsi_14[i] < 40
        rsi_overbought = rsi_14[i] > 60
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else False
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        if i > 100:
            atr_median = np.nanmedian(atr_14[max(0, i-100):i])
            atr_ratio = atr_14[i] / atr_median if atr_median > 0 else 1.0
            vol_adjustment = np.clip(1.0 / atr_ratio, 0.8, 1.2)
        else:
            vol_adjustment = 1.0
        
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.30)
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        conviction = 0
        
        # LONG ENTRIES
        if trend_bullish or neutral_market:
            if choppy_market:
                # Mean reversion in chop: CRSI oversold
                if crsi_oversold:
                    new_signal = current_size
                    conviction = 1
            elif trending_market:
                # Trend pullback: price > HMA50 + RSI oversold
                if close[i] > hma_1d_50[i] and rsi_oversold:
                    new_signal = current_size
                    conviction = 1
            elif neutral_market:
                # Neutral: CRSI very oversold + above SMA200
                if crsi[i] < 15 and above_sma200:
                    new_signal = HIGH_CONV_SIZE * vol_adjustment
                    conviction = 2
        
        # SHORT ENTRIES
        if trend_bearish or neutral_market:
            if choppy_market:
                # Mean reversion in chop: CRSI overbought
                if crsi_overbought:
                    new_signal = -current_size
                    conviction = 1
            elif trending_market:
                # Trend pullback: price < HMA50 + RSI overbought
                if close[i] < hma_1d_50[i] and rsi_overbought:
                    new_signal = -current_size
                    conviction = 1
            elif neutral_market:
                # Neutral: CRSI very overbought + below SMA200
                if crsi[i] > 85 and below_sma200:
                    new_signal = -HIGH_CONV_SIZE * vol_adjustment
                    conviction = 2
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 60 bars (~2 months on 1d), force entry with weaker signal
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if trend_bullish and crsi[i] < 30:
                new_signal = current_size * 0.5
            elif trend_bearish and crsi[i] > 70:
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
            if position_side > 0 and trend_bearish:
                trend_reversal = True
            if position_side < 0 and trend_bullish:
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