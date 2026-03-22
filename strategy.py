#!/usr/bin/env python3
"""
Experiment #087: 1d Primary + 1w HTF — KAMA Adaptive Trend + Choppiness Regime + Connors RSI

Hypothesis: The successful 12h strategy (#076) can be adapted to 1d timeframe with 
better results because daily bars filter out noise while still capturing major moves.
Key improvements over #076:
1. KAMA instead of HMA - adapts to market efficiency, reduces whipsaws in chop
2. 1w HTF instead of 1d - stronger major trend filter for daily entries
3. Looser CRSI thresholds - ensure enough trades on 1d (target 15-40/year)
4. Add volume confirmation - entries only on above-average volume days
5. Asymmetric sizing - larger positions in confirmed trend regime

Why this should work:
- 1d timeframe naturally limits trades to avoid fee drag
- KAMA adapts speed based on volatility (fast in trends, slow in chop)
- 1w HMA slope provides strong directional bias (prevents counter-trend trades)
- Choppiness regime detection proven in #076 to work well
- Connors RSI has 75% win rate for mean reversion entries
- Volume filter ensures entries on conviction days

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
Target trades: 15-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_kama_chop_connors_1w_v1"
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

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
    return chop

def calculate_connors_rsi(close, rsi_period=3, streak_period=2, rank_period=100):
    """
    Calculate Connors RSI (CRSI).
    CRSI = (RSI(3) + RSI_Streak(2) + PercentRank(100)) / 3
    """
    close_s = pd.Series(close)
    
    rsi_3 = calculate_rsi(close, rsi_period)
    
    delta = close_s.diff()
    streak = np.zeros(len(close))
    
    for i in range(1, len(close)):
        if delta.iloc[i] > 0:
            streak[i] = streak[i-1] + 1 if streak[i-1] >= 0 else 1
        elif delta.iloc[i] < 0:
            streak[i] = streak[i-1] - 1 if streak[i-1] <= 0 else -1
        else:
            streak[i] = 0
    
    streak_rsi = np.zeros(len(close))
    for i in range(len(close)):
        if streak[i] >= 0:
            streak_rsi[i] = min(100, 50 + streak[i] * 10)
        else:
            streak_rsi[i] = max(0, 50 + streak[i] * 10)
    
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).
    KAMA adapts to market efficiency - fast in trends, slow in chop.
    """
    close_s = pd.Series(close)
    
    # Efficiency Ratio (ER)
    change = close_s.diff(er_period).abs()
    volatility = close_s.diff().abs().rolling(window=er_period, min_periods=er_period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    
    # Smoothing constant
    fast_sc = 2 / (fast_period + 1)
    slow_sc = 2 / (slow_period + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros(len(close))
    kama[0] = close[0]
    
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    return kama

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope over lookback period."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_volume_ratio(volume, period=20):
    """Calculate volume ratio vs rolling average."""
    vol_s = pd.Series(volume)
    vol_avg = vol_s.rolling(window=period, min_periods=period).mean()
    vol_ratio = vol_s / vol_avg.replace(0, np.nan)
    vol_ratio = vol_ratio.fillna(1.0).values
    return vol_ratio

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    vol_ratio = calculate_volume_ratio(volume, 20)
    
    # KAMA for adaptive trend
    kama_21 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    kama_50 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    
    # Also calculate HMA for comparison
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    
    signals = np.zeros(n)
    
    # Base position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(kama_21[i]) or np.isnan(kama_50[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # HMA slope > 0.3 = bullish bias (prefer longs)
        # HMA slope < -0.3 = bearish bias (prefer shorts)
        trend_1w_bullish = hma_1w_slope_aligned[i] > 0.3
        trend_1w_bearish = hma_1w_slope_aligned[i] < -0.3
        trend_1w_neutral = not trend_1w_bullish and not trend_1w_bearish
        
        # Price vs 1w HMA for additional confirmation
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 55 = range market (mean revert strategy)
        # CHOP < 45 = trend market (trend follow strategy)
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === CONNORS RSI SIGNALS (looser thresholds for more trades) ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_moderate_low = crsi[i] < 40
        crsi_moderate_high = crsi[i] > 60
        
        # === 1D TREND CONFIRMATION ===
        kama_bullish = kama_21[i] > kama_50[i]
        kama_bearish = kama_21[i] < kama_50[i]
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === VOLUME CONFIRMATION ===
        volume_confirmed = vol_ratio[i] > 0.8  # At least 80% of avg volume
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in neutral 1w trend or transitional chop
        if trend_1w_neutral or (not is_range_market and not is_trend_market):
            current_size = REDUCED_SIZE
        
        # Increase size in strong trend regime with confirmation
        if is_trend_market and trend_1w_bullish and kama_bullish:
            current_size = BASE_SIZE
        if is_trend_market and trend_1w_bearish and kama_bearish:
            current_size = BASE_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_confidence = 0
        
        if is_range_market:
            # Mean reversion in range: buy oversold
            if crsi_oversold and volume_confirmed:
                long_confidence += 1
            if trend_1w_bullish or price_above_1w_hma:
                long_confidence += 1
            if kama_bullish or hma_bullish:
                long_confidence += 1
            
            if long_confidence >= 2:
                new_signal = current_size
        elif is_trend_market:
            # Trend following: buy pullback in uptrend
            if trend_1w_bullish and (kama_bullish or hma_bullish):
                if crsi_moderate_low and volume_confirmed:
                    new_signal = current_size
                elif crsi[i] < 45:
                    new_signal = current_size * 0.8
            elif price_above_1w_hma and (kama_bullish or hma_bullish):
                if crsi[i] < 45 and volume_confirmed:
                    new_signal = current_size * 0.8
        else:
            # Transitional: weaker signals
            if (trend_1w_bullish or price_above_1w_hma) and crsi_moderate_low:
                new_signal = REDUCED_SIZE * 0.8
        
        # SHORT ENTRIES
        short_confidence = 0
        
        if is_range_market:
            # Mean reversion in range: sell overbought
            if crsi_overbought and volume_confirmed:
                short_confidence += 1
            if trend_1w_bearish or price_below_1w_hma:
                short_confidence += 1
            if kama_bearish or hma_bearish:
                short_confidence += 1
            
            if short_confidence >= 2:
                new_signal = -current_size
        elif is_trend_market:
            # Trend following: sell pullback in downtrend
            if trend_1w_bearish and (kama_bearish or hma_bearish):
                if crsi_moderate_high and volume_confirmed:
                    new_signal = -current_size
                elif crsi[i] > 55:
                    new_signal = -current_size * 0.8
            elif price_below_1w_hma and (kama_bearish or hma_bearish):
                if crsi[i] > 55 and volume_confirmed:
                    new_signal = -current_size * 0.8
        else:
            # Transitional: weaker signals
            if (trend_1w_bearish or price_below_1w_hma) and crsi_moderate_high:
                new_signal = -REDUCED_SIZE * 0.8
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 90 bars (~90 days on 1d), allow weaker entry
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and crsi[i] < 45:
                new_signal = REDUCED_SIZE * 0.6
            elif trend_1w_bearish and crsi[i] > 55:
                new_signal = -REDUCED_SIZE * 0.6
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if market becomes strongly trending bearish
            if position_side > 0 and is_trend_market and trend_1w_bearish and kama_bearish:
                regime_reversal = True
            # Exit short if market becomes strongly trending bullish
            if position_side < 0 and is_trend_market and trend_1w_bullish and kama_bullish:
                regime_reversal = True
        
        # Apply stoploss or regime reversal
        if stoploss_triggered or regime_reversal:
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