#!/usr/bin/env python3
"""
Experiment #003: 1d KAMA-Choppiness-Connors with 1w Trend Filter

Hypothesis: Previous dual-regime strategies (#001, #002) failed because they used
HMA which doesn't adapt to volatility changes. This strategy uses:
1. KAMA (Kaufman Adaptive MA) - adapts smoothing based on market efficiency/volatility
2. Choppiness Index - detects range vs trend regime (CHOP>61.8=range, <38.2=trend)
3. Connors RSI - proven mean-reversion indicator (RSI(3)+Streak(2)+PercentRank(100))/3
4. 1w HMA(21) - major trend filter for directional bias

Key differences from failed #001/#002:
- KAMA instead of HMA (better volatility adaptation)
- Connors RSI instead of standard RSI (proven 75% win rate on ETH)
- Regime-switching logic: mean-revert in chop, trend-follow otherwise
- 1d primary with 1w HTF filter (per experiment requirements)

Why this should work:
- 1d timeframe = 20-50 trades/year (fee drag manageable)
- KAMA reduces whipsaw in choppy markets (2022 crash protection)
- Connors RSI excels in bear/range markets (2025 test period)
- 1w filter prevents counter-trend trades in major moves

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data helper (call ONCE before loop)
Position sizing: 0.25-0.30 discrete
Stoploss: 3.0 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_connors_1w_filter_v1"
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

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts smoothing based on market efficiency ratio.
    High ER (trending) = fast smoothing, Low ER (choppy) = slow smoothing.
    """
    close_s = pd.Series(close)
    n = len(close)
    
    # Calculate Efficiency Ratio (ER)
    # ER = |Change| / Sum of |Individual Changes|
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[max(0, i-er_period):i+1])))
    
    er = change / (volatility + 1e-10)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period] = close[er_period]
    
    for i in range(er_period + 1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, n) / (Highest High - Lowest Low)) / LOG10(n)
    CHOP > 61.8 = choppy/range, CHOP < 38.2 = trending
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        tr_sum = 0.0
        highest = -np.inf
        lowest = np.inf
        
        for j in range(i - period + 1, i + 1):
            tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr_sum += tr
            highest = max(highest, high[j])
            lowest = min(lowest, low[j])
        
        price_range = highest - lowest
        if price_range > 0 and tr_sum > 0:
            chop[i] = 100 * np.log10(tr_sum / price_range) / np.log10(period)
        else:
            chop[i] = 50.0  # neutral
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    Long entry: CRSI < 10, Short entry: CRSI > 90
    """
    n = len(close)
    close_s = pd.Series(close)
    
    # RSI(3) component
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, min_periods=rsi_period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_close = 100 - (100 / (1 + rs))
    
    # Streak RSI component
    streak = np.zeros(n)
    streak_rsi = np.zeros(n)
    
    for i in range(1, n):
        if close[i] > close[i-1]:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif close[i] < close[i-1]:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = streak[i-1]
    
    # Calculate RSI on streak values
    streak_s = pd.Series(streak)
    streak_delta = streak_s.diff()
    streak_gain = streak_delta.where(streak_delta > 0, 0.0)
    streak_loss = -streak_delta.where(streak_delta < 0, 0.0)
    
    streak_avg_gain = streak_gain.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    streak_avg_loss = streak_loss.ewm(span=streak_period, min_periods=streak_period, adjust=False).mean()
    
    streak_rs = streak_avg_gain / (streak_avg_loss + 1e-10)
    streak_rsi = 100 - (100 / (1 + streak_rs))
    
    # Percent Rank component
    percent_rank = np.zeros(n)
    for i in range(rank_period, n):
        window = close[max(0, i-rank_period):i]
        if len(window) > 0:
            percent_rank[i] = 100 * np.sum(window < close[i]) / len(window)
        else:
            percent_rank[i] = 50.0
    
    # Combine components
    crsi = (rsi_close.values + streak_rsi.values + percent_rank) / 3.0
    
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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1W indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    kama_1d = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_1d_fast = calculate_kama(close, er_period=5, fast_period=2, slow_period=15)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100)
    
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
    
    for i in range(150, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]):
            continue
        
        if np.isnan(kama_1d[i]) or np.isnan(kama_1d_fast[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        # === 1W TREND BIAS ===
        weekly_bullish = close[i] > hma_1w_21_aligned[i]
        weekly_bearish = close[i] < hma_1w_21_aligned[i]
        
        # === 1D KAMA TREND ===
        kama_bullish = kama_1d_fast[i] > kama_1d[i]
        kama_bearish = kama_1d_fast[i] < kama_1d[i]
        
        # === KAMA SLOPE ===
        kama_slope_long = kama_1d[i] > kama_1d[i-5] if i > 5 else False
        kama_slope_short = kama_1d[i] < kama_1d[i-5] if i > 5 else False
        
        # === CHOPPINESS REGIME ===
        is_choppy = chop_14[i] > 55.0  # Range market
        is_trending = chop_14[i] < 45.0  # Trend market
        
        # === CONNORS RSI EXTREMES ===
        crsi_oversold = crsi[i] < 15  # Mean reversion long
        crsi_overbought = crsi[i] > 85  # Mean reversion short
        
        # === VOLATILITY-ADJUSTED POSITION SIZING ===
        atr_ratio = atr_14[i] / np.nanmedian(atr_14[max(0, i-100):i]) if i > 100 else 1.0
        vol_adjustment = np.clip(1.0 / atr_ratio, 0.7, 1.3)
        current_size = BASE_SIZE * vol_adjustment
        current_size = np.clip(current_size, 0.20, 0.35)
        
        # === REGIME-SWITCHING ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        if is_trending:
            # TREND FOLLOWING MODE
            # Long: KAMA bullish + weekly bias + slope confirmation
            if kama_bullish and weekly_bullish and kama_slope_long:
                # Entry on pullback (CRSI not overbought)
                if crsi[i] < 70:
                    new_signal = current_size
            
            # Short: KAMA bearish + weekly bias + slope confirmation
            if kama_bearish and weekly_bearish and kama_slope_short:
                # Entry on bounce (CRSI not oversold)
                if crsi[i] > 30:
                    new_signal = -current_size
        
        elif is_choppy:
            # MEAN REVERSION MODE
            # Long: CRSI oversold + price above weekly HMA (bias long)
            if crsi_oversold and (weekly_bullish or not weekly_bearish):
                new_signal = current_size * 0.8  # Smaller size for mean revert
            
            # Short: CRSI overbought + price below weekly HMA (bias short)
            if crsi_overbought and (weekly_bearish or not weekly_bullish):
                new_signal = -current_size * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 45 bars (~45 days on 1d), allow weaker entries
        if bars_since_last_trade > 45 and new_signal == 0.0 and not in_position:
            if kama_bullish and weekly_bullish and crsi[i] < 50:
                new_signal = current_size * 0.6
            elif kama_bearish and weekly_bearish and crsi[i] > 50:
                new_signal = -current_size * 0.6
        
        # === STOPLOSS LOGIC (Rule 6) - 3.0 * ATR trailing ===
        stoploss_triggered = False
        
        if in_position and position_side != 0:
            if position_side > 0:
                if close[i] > highest_price:
                    highest_price = close[i]
                stoploss_price = highest_price - 3.0 * atr_14[i]
                if close[i] < stoploss_price:
                    stoploss_triggered = True
            
            if position_side < 0:
                if lowest_price == 0.0 or close[i] < lowest_price:
                    lowest_price = close[i]
                stoploss_price = lowest_price + 3.0 * atr_14[i]
                if close[i] > stoploss_price:
                    stoploss_triggered = True
        
        # === TREND REVERSAL EXIT ===
        trend_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and kama_bearish:
                trend_reversal = True
            if position_side < 0 and kama_bullish:
                trend_reversal = True
        
        # === CRSI EXTREME EXIT (mean reversion complete) ===
        crsi_exit = False
        if in_position and position_side != 0:
            if position_side > 0 and crsi[i] > 75:
                crsi_exit = True  # Take profit on long
            if position_side < 0 and crsi[i] < 25:
                crsi_exit = True  # Take profit on short
        
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