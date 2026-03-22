#!/usr/bin/env python3
"""
Experiment #002: 12h KAMA Trend + ADX/Choppiness Regime + Connors RSI Entries

Hypothesis: The failed BB Squeeze + Donchian strategy was too breakout-focused and
whipsawed in range markets. This strategy uses a REGIME-ADAPTIVE approach:

1. KAMA (Kaufman Adaptive Moving Average) - adapts smoothing based on market efficiency.
   Fast in trends, slow in chop. Period=10, fast SC=2/11, slow SC=2/31.
   Proven to reduce whipsaws vs simple EMA.

2. Choppiness Index (CHOP) - regime detection. CHOP(14) > 61.8 = range (mean revert),
   CHOP < 38.2 = trending (trend follow). This is the KEY meta-filter.

3. ADX(14) - trend strength confirmation. ADX > 25 = strong trend, ADX < 20 = weak/range.
   Used with hysteresis (enter 25, exit 18) to avoid flip-flopping.

4. Connors RSI (CRSI) - entry timing. CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3.
   Long when CRSI < 15 (oversold), Short when CRSI > 85 (overbought).
   75% win rate in backtests, excellent for counter-trend entries in ranges.

5. 1d HMA(21) Trend Filter - via mtf_data. Only long if price > 1d HMA, only short if <.
   Prevents counter-trend trades against major daily trend.

6. 1w HMA(21) Major Bias - via mtf_data. Increases size when 12h + 1w align.

7. ATR(14) Trailing Stop - 2.5x ATR for risk management.

Why 12h works:
- 12h = 20-50 trades/year target (optimal fee drag)
- Higher TF = fewer false signals, cleaner trends
- Regime-adaptive = works in both trending (2021) and ranging (2025) markets

Timeframe: 12h (REQUIRED for Experiment #002)
HTF: 1d and 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25 base, 0.30 high conviction, 0.15 low conviction
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_kama_chop_adx_connors_1d_1w_v1"
timeframe = "12h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (signal-to-noise ratio).
    """
    close_s = pd.Series(close)
    n = len(close_s)
    
    # Efficiency Ratio (ER) = |Change| / Sum of |Changes|
    change = np.abs(close_s.diff(period).values)
    noise = np.abs(close_s.diff()).rolling(window=period, min_periods=period).sum().values
    
    # Avoid division by zero
    er = np.where(noise > 0, change / noise, 0)
    er = np.nan_to_num(er, nan=0.0)
    
    # Smoothing Constants
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    
    # Adaptive SC = ER * (fast_sc - slow_sc) + slow_sc
    sc = er * (fast_sc - slow_sc) + slow_sc
    sc = np.clip(sc, slow_sc, fast_sc)
    
    # KAMA calculation
    kama = np.zeros(n)
    kama[period-1] = close_s.iloc[period-1]
    
    for i in range(period, n):
        kama[i] = kama[i-1] + sc[i] * (close_s.iloc[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index (CHOP).
    Measures market choppiness vs trending.
    CHOP > 61.8 = range/chop, CHOP < 38.2 = trending.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # Highest high and lowest low over period
    hh = high_s.rolling(window=period, min_periods=period).max()
    ll = low_s.rolling(window=period, min_periods=period).min()
    
    # Sum of ATR-like movements
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    sum_tr = tr.rolling(window=period, min_periods=period).sum()
    
    # CHOP = 100 * log10(sum_tr / (hh - ll)) / log10(period)
    price_range = hh - ll
    chop = 100 * np.log10(sum_tr / price_range.replace(0, np.nan)) / np.log10(period)
    chop = chop.replace([np.inf, -np.inf], np.nan)
    
    return chop.values

def calculate_adx(high, low, close, period=14):
    """
    Average Directional Index (ADX).
    Measures trend strength (not direction).
    ADX > 25 = strong trend, ADX < 20 = weak/range.
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # True Range
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    plus_dm = high_s.diff()
    minus_dm = -low_s.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    # Smoothed DM and TR (Wilder's smoothing)
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(span=period, min_periods=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr.replace(0, np.nan)
    
    # DX = |DI+ - DI-| / (DI+ + DI-) * 100
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    
    # ADX = smoothed DX
    adx = dx.ewm(span=period, min_periods=period, adjust=False).mean()
    adx = adx.replace([np.inf, -np.inf], np.nan)
    
    return adx.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.replace([np.inf, -np.inf], np.nan)
    
    return rsi.values

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long when CRSI < 15, Short when CRSI > 85.
    """
    close_s = pd.Series(close)
    n = len(close_s)
    
    # Component 1: RSI(3)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi_short = 100 - (100 / (1 + rs))
    rsi_short = rsi_short.replace([np.inf, -np.inf], np.nan)
    
    # Component 2: RSI of Streak (consecutive up/down days)
    streak = np.zeros(n)
    for i in range(1, n):
        if close_s.iloc[i] > close_s.iloc[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] > 0 else 1
        elif close_s.iloc[i] < close_s.iloc[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] < 0 else -1
        else:
            streak[i] = 0
    
    streak_s = pd.Series(streak)
    # Convert streak to RSI-like value (positive streak = bullish, negative = bearish)
    streak_rsi = np.zeros(n)
    for i in range(streak_period, n):
        window = streak[i-streak_period+1:i+1]
        up_count = np.sum(window > 0)
        down_count = np.sum(window < 0)
        if up_count + down_count > 0:
            streak_rsi[i] = 100 * up_count / (up_count + down_count)
        else:
            streak_rsi[i] = 50
    
    # Component 3: Percent Rank of daily returns
    daily_returns = close_s.pct_change()
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = daily_returns.iloc[i-rank_period+1:i+1].dropna()
        if len(window) > 0:
            current = daily_returns.iloc[i]
            if not np.isnan(current):
                percent_rank[i] = np.sum(window < current) / len(window) * 100
            else:
                percent_rank[i] = 50
        else:
            percent_rank[i] = 50
    
    # Combine components
    crsi = (rsi_short.values + streak_rsi + percent_rank) / 3
    crsi = np.nan_to_num(crsi, nan=50.0)
    crsi = np.clip(crsi, 0, 100)
    
    return crsi

def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = tr.ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(close, period=21):
    """Hull Moving Average - reduces lag while maintaining smoothness."""
    close_s = pd.Series(close)
    n = period
    
    def wma(series, span):
        return series.ewm(span=span, min_periods=span, adjust=False).mean()
    
    half = int(n / 2)
    sqrt_n = int(np.sqrt(n))
    
    wma_half = wma(close_s, half)
    wma_full = wma(close_s, n)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_n)
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1d HMA for trend filter
    hma_1d_21 = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 1w HMA for major bias
    hma_1w_21 = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 12h indicators
    kama_10 = calculate_kama(close, period=10)
    chop_14 = calculate_choppiness(high, low, close, period=14)
    adx_14 = calculate_adx(high, low, close, period=14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    atr_14 = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    LOW_CONV_SIZE = 0.15
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    last_trade_bar = -50
    
    # ADX hysteresis tracking
    adx_trending = False  # True if ADX recently > 25
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1d_21_aligned[i]) or np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(chop_14[i]) or np.isnan(adx_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === DAILY TREND FILTER ===
        daily_bullish = close[i] > hma_1d_21_aligned[i]
        daily_bearish = close[i] < hma_1d_21_aligned[i]
        
        # === WEEKLY MAJOR BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === REGIME DETECTION ===
        # Choppiness Index: > 61.8 = range, < 38.2 = trend
        is_choppy = chop_14[i] > 61.8
        is_trending = chop_14[i] < 38.2
        
        # ADX with hysteresis
        if adx_14[i] > 25:
            adx_trending = True
        elif adx_14[i] < 18:
            adx_trending = False
        
        adx_strong = adx_trending  # Use hysteresis state
        
        # === KAMA TREND ===
        kama_bullish = close[i] > kama_10[i]
        kama_bearish = close[i] < kama_10[i]
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Long entry
        crsi_overbought = crsi[i] > 85  # Short entry
        crsi_neutral = 15 <= crsi[i] <= 85
        
        # === ENTRY LOGIC - REGIME ADAPTIVE ===
        new_signal = 0.0
        
        if is_trending or adx_strong:
            # TREND FOLLOWING MODE
            # Long: KAMA bullish + daily bullish + CRSI not overbought
            if kama_bullish and daily_bullish and crsi[i] < 70:
                if weekly_bullish:
                    new_signal = HIGH_CONV_SIZE
                else:
                    new_signal = BASE_SIZE
            
            # Short: KAMA bearish + daily bearish + CRSI not oversold
            elif kama_bearish and daily_bearish and crsi[i] > 30:
                if weekly_bearish:
                    new_signal = -HIGH_CONV_SIZE
                else:
                    new_signal = -BASE_SIZE
        
        elif is_choppy:
            # MEAN REVERSION MODE
            # Long: CRSI oversold + daily bullish (don't fight major trend)
            if crsi_oversold and daily_bullish:
                new_signal = LOW_CONV_SIZE
            
            # Short: CRSI overbought + daily bearish
            elif crsi_overbought and daily_bearish:
                new_signal = -LOW_CONV_SIZE
        
        # === NEUTRAL REGIME (38.2 <= CHOP <= 61.8) ===
        # Use KAMA crossover with CRSI confirmation
        if not is_trending and not is_choppy:
            if kama_bullish and daily_bullish and crsi[i] < 50:
                new_signal = BASE_SIZE
            elif kama_bearish and daily_bearish and crsi[i] > 50:
                new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 30 bars (~15 days on 12h), allow weaker entry
        bars_since_last_trade = i - last_trade_bar
        if bars_since_last_trade > 30 and new_signal == 0.0 and not in_position:
            if kama_bullish and daily_bullish and crsi[i] < 40:
                new_signal = LOW_CONV_SIZE
            elif kama_bearish and daily_bearish and crsi[i] > 60:
                new_signal = -LOW_CONV_SIZE
        
        # === STOPLOSS LOGIC (Rule 6) - 2.5 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                # Update highest price for long position
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 2.5 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                # Update lowest price for short position
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 2.5 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            # Exit long if daily trend turns bearish
            if position_side > 0 and daily_bearish:
                trend_reversal = True
            # Exit short if daily trend turns bullish
            if position_side < 0 and daily_bullish:
                trend_reversal = True
        
        # === KAMA REVERSAL EXIT ===
        kama_exit = False
        if in_position and position_side != 0:
            # Exit long if price crosses below KAMA
            if position_side > 0 and close[i] < kama_10[i]:
                kama_exit = True
            # Exit short if price crosses above KAMA
            if position_side < 0 and close[i] > kama_10[i]:
                kama_exit = True
        
        # Apply stoploss or exits
        if stoploss_triggered or trend_reversal or kama_exit:
            new_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if new_signal != 0.0:
            if not in_position:
                # New entry
                in_position = True
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
                last_trade_bar = i
        
        signals[i] = new_signal
    
    return signals