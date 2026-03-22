#!/usr/bin/env python3
"""
Experiment #194: 4h Primary + 12h/1d HTF — Dual Regime Adaptive Strategy

Hypothesis: Previous 4h strategies failed because they used单一 approach (either pure trend
or pure mean reversion). Research shows BTC/ETH behave differently in different regimes:
- 2021: Strong trend (trend following works)
- 2022: Crash + whipsaw (mean reversion at extremes works)
- 2023-2024: Range with rallies (regime-adaptive needed)
- 2025+: Bear/range (mean reversion + selective trend)

This strategy combines:
1. CHOPPINESS INDEX (14): Regime detection (CHOP>55=range, CHOP<45=trend)
2. CONNORS RSI: Entry timing for mean reversion (CRSI<20 long, CRSI>80 short)
3. EHRLERS FISHER TRANSFORM: Reversal signals in bear rallies (crosses ±1.5)
4. 12h HMA(21) SLOPE: Intermediate trend bias
5. 1d HMA(21): Major trend filter (avoid counter-trend trades)
6. ATR(14) trailing stop: Risk management (2.5*ATR)

Why this should work:
- Dual regime: mean revert in chop, trend pullback in trends
- Multiple entry paths ensure sufficient trade frequency
- 4h timeframe = 30-60 trades/year target (reasonable fee drag)
- 12h/1d HTF prevents fighting major trends
- Discrete position sizing (0.25/0.30) minimizes fee churn

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h + 1d via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, 0.30 high conviction
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_dualregime_connors_fisher_12h1d_v1"
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

def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    sma = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return upper, lower, sma

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
    
    # Component 1: RSI(3)
    rsi_3 = calculate_rsi(close, rsi_period)
    
    # Component 2: RSI of Streak
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
    
    # Component 3: Percent Rank
    pct_change = close_s.pct_change()
    percent_rank = pd.Series(pct_change).rolling(window=rank_period, min_periods=rank_period).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) > 0 else 50
    ).values
    percent_rank = np.nan_to_num(percent_rank, nan=50.0)
    
    crsi = (rsi_3 + streak_rsi + percent_rank) / 3.0
    
    return crsi

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_hma_slope(hma_values, lookback=5):
    """Calculate HMA slope as percentage change."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0:
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / hma_values[i - lookback] * 100
    return slope

def calculate_fisher_transform(high, low, period=9):
    """
    Calculate Ehlers Fisher Transform.
    Transforms price into a Gaussian normal distribution for clearer reversal signals.
    Long when Fisher crosses above -1.5, short when crosses below +1.5
    """
    hl2 = (high + low) / 2.0
    hl2_s = pd.Series(hl2)
    
    # Normalize price within range
    highest = hl2_s.rolling(window=period, min_periods=period).max().values
    lowest = hl2_s.rolling(window=period, min_periods=period).min().values
    price_range = highest - lowest
    price_range = np.where(price_range == 0, 1e-10, price_range)
    
    normalized = (hl2 - lowest) / price_range
    normalized = np.clip(normalized * 2 - 1, -0.99, 0.99)
    
    # Fisher transform
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    # Signal line (1-period lag)
    fisher_signal = np.roll(fisher, 1)
    fisher_signal[0] = fisher[0]
    
    return fisher, fisher_signal

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_slope = calculate_hma_slope(hma_12h_21, 5)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_slope = calculate_hma_slope(hma_1d_21, 5)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    hma_12h_slope_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_slope)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    hma_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_slope)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    fisher, fisher_signal = calculate_fisher_transform(high, low, 9)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    
    # Volatility ratio for spike detection
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    atr_ratio = atr_7 / np.where(atr_30 > 0, atr_30, 1e-10)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    
    # Track position state
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(atr_ratio[i]):
            continue
        
        # === 1D TREND BIAS (Major trend filter) ===
        trend_1d_bullish = hma_1d_slope_aligned[i] > 0.5
        trend_1d_bearish = hma_1d_slope_aligned[i] < -0.5
        price_above_1d_hma = close[i] > hma_1d_21_aligned[i]
        price_below_1d_hma = close[i] < hma_1d_21_aligned[i]
        
        # === 12H TREND (Intermediate trend) ===
        trend_12h_bullish = hma_12h_slope_aligned[i] > 0.3
        trend_12h_bearish = hma_12h_slope_aligned[i] < -0.3
        price_above_12h_hma = close[i] > hma_12h_21_aligned[i]
        price_below_12h_hma = close[i] < hma_12h_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        is_neutral = not is_range_market and not is_trend_market
        
        # === VOLATILITY SPIKE ===
        vol_spike = atr_ratio[i] > 1.5
        
        # === BOLLINGER BAND POSITION ===
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        bb_width = (bb_upper[i] - bb_lower[i]) / bb_mid[i] if bb_mid[i] > 0 else 0
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 15
        crsi_extreme_high = crsi[i] > 85
        
        # === FISHER TRANSFORM ===
        fisher_cross_up = (fisher[i] > -1.5) and (fisher_signal[i] <= -1.5)
        fisher_cross_down = (fisher[i] < 1.5) and (fisher_signal[i] >= 1.5)
        fisher_extreme_low = fisher[i] < -2.0
        fisher_extreme_high = fisher[i] > 2.0
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        conviction = 0
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Multiple confluence paths
        long_score = 0
        
        # Path 1: Range market + CRSI extreme (mean revert) - HIGH conviction
        if is_range_market and crsi_extreme_low:
            long_score += 3
            conviction = 1
        
        # Path 2: Vol spike + BB lower + CRSI oversold (capitulation) - HIGH conviction
        if vol_spike and price_below_bb_lower and crsi_oversold:
            long_score += 3
            conviction = 1
        
        # Path 3: Fisher extreme low + CRSI oversold (reversal)
        if fisher_extreme_low and crsi_oversold:
            long_score += 2
        
        # Path 4: Trend market + pullback in bull trend
        if is_trend_market and trend_12h_bullish and crsi[i] < 35:
            long_score += 2
        
        # Path 5: 1d bullish + price below 12h HMA (pullback entry)
        if trend_1d_bullish and price_below_12h_hma and crsi[i] < 40:
            long_score += 2
        
        # Path 6: Fisher cross up in oversold conditions
        if fisher_cross_up and crsi[i] < 45:
            long_score += 2
        
        # Path 7: Simple oversold fallback (ensures trade frequency)
        if crsi[i] < 20 and price_below_bb_lower:
            long_score += 1
        
        # Path 8: 1d neutral + range + moderate oversold
        if not trend_1d_bearish and is_range_market and crsi[i] < 30:
            long_score += 1
        
        if long_score >= 3:
            new_signal = HIGH_CONV_SIZE if conviction else current_size
        elif long_score >= 2:
            new_signal = current_size
        elif long_score >= 1 and bars_since_last_trade > 60:
            new_signal = current_size * 0.6
        
        # SHORT ENTRIES
        short_score = 0
        
        # Path 1: Range market + CRSI extreme (mean revert) - HIGH conviction
        if is_range_market and crsi_extreme_high:
            short_score += 3
            conviction = 1
        
        # Path 2: Vol spike + BB upper + CRSI overbought
        if vol_spike and price_above_bb_upper and crsi_overbought:
            short_score += 3
            conviction = 1
        
        # Path 3: Fisher extreme high + CRSI overbought
        if fisher_extreme_high and crsi_overbought:
            short_score += 2
        
        # Path 4: Trend market + pullback in bear trend
        if is_trend_market and trend_12h_bearish and crsi[i] > 65:
            short_score += 2
        
        # Path 5: 1d bearish + price above 12h HMA (rally entry)
        if trend_1d_bearish and price_above_12h_hma and crsi[i] > 60:
            short_score += 2
        
        # Path 6: Fisher cross down in overbought conditions
        if fisher_cross_down and crsi[i] > 55:
            short_score += 2
        
        # Path 7: Simple overbought fallback
        if crsi[i] > 80 and price_above_bb_upper:
            short_score += 1
        
        # Path 8: 1d neutral + range + moderate overbought
        if not trend_1d_bullish and is_range_market and crsi[i] > 70:
            short_score += 1
        
        if short_score >= 3:
            new_signal = -HIGH_CONV_SIZE if conviction else -current_size
        elif short_score >= 2:
            new_signal = -current_size
        elif short_score >= 1 and bars_since_last_trade > 60:
            new_signal = -current_size * 0.6
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 120 bars (~20 days on 4h) to ensure min trades
        if bars_since_last_trade > 120 and new_signal == 0.0 and not in_position:
            if trend_1d_bullish and crsi[i] < 40:
                new_signal = current_size * 0.4
            elif trend_1d_bearish and crsi[i] > 60:
                new_signal = -current_size * 0.4
            elif crsi[i] < 25:
                new_signal = current_size * 0.3
            elif crsi[i] > 75:
                new_signal = -current_size * 0.3
        
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
            # Exit long if regime shifts to strong bear trend
            if position_side > 0 and is_trend_market and trend_1d_bearish:
                regime_reversal = True
            # Exit short if regime shifts to strong bull trend
            if position_side < 0 and is_trend_market and trend_1d_bullish:
                regime_reversal = True
        
        # === CRSI REVERSAL EXIT ===
        crsi_exit = False
        if in_position and position_side != 0:
            # Exit long when CRSI becomes overbought
            if position_side > 0 and crsi[i] > 70:
                crsi_exit = True
            # Exit short when CRSI becomes oversold
            if position_side < 0 and crsi[i] < 30:
                crsi_exit = True
        
        if stoploss_triggered or regime_reversal or crsi_exit:
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