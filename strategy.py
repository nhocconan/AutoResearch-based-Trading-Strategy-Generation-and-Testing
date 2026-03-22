#!/usr/bin/env python3
"""
Experiment #213: 1d Primary + 1w HTF — Regime-Adaptive Mean Reversion + Trend

Hypothesis: Previous 12h strategies failed because they were too complex with too many
conflicting filters. For 1d timeframe, we need SIMPLER logic with fewer conditions that
actually trigger. Research shows BTC/ETH perform best with:
1. Regime-adaptive logic (mean revert in chop, trend follow in trends)
2. Weekly HMA for major trend bias (avoid counter-trend in strong moves)
3. Connors RSI for precise entry timing (75% win rate in literature)
4. ATR-based stoploss (2.5*ATR) to limit drawdown
5. Asymmetric sizing (larger positions when regime + trend align)

Why 1d should work:
- 10-30 trades/year target = minimal fee drag
- Weekly HTF captures major crypto cycles (bull/bear markets last months)
- Daily bars filter out noise that killed lower TF strategies
- Simpler entry conditions = more trades (avoid 0-trade auto-reject)

Key differences from failed strategies:
- Fewer confluence requirements (2-3 conditions max, not 5+)
- Lower CRSI thresholds (20/80 instead of 15/85) for more triggers
- Frequency safeguard: force entry if no trade in 60 days
- Discrete sizing: 0.0, ±0.25, ±0.30 (minimize churn)

Timeframe: 1d (REQUIRED)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.25 base, 0.30 max (discrete levels)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 15-40/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_regime_connors_hma_1w_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HTF indicators
    hma_1w_21 = calculate_hma(df_1w['close'].values, 21)
    hma_1w_slope = calculate_hma_slope(hma_1w_21, 3)
    
    # Align HTF to LTF (Rule 2 - auto shift(1))
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    bb_upper, bb_lower, bb_mid = calculate_bollinger_bands(close, 20, 2.0)
    chop_14 = calculate_choppiness(high, low, close, 14)
    crsi = calculate_connors_rsi(close, 3, 2, 100)
    rsi_14 = calculate_rsi(close, 14)
    
    # Price position within BB
    bb_width = bb_upper - bb_lower
    bb_width = np.where(bb_width == 0, 1e-10, bb_width)
    bb_position = (close - bb_lower) / bb_width
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete, max 0.35)
    BASE_SIZE = 0.25
    MAX_SIZE = 0.30
    
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
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(crsi[i]):
            continue
        
        if np.isnan(bb_lower[i]) or np.isnan(bb_position[i]):
            continue
        
        # === 1W TREND BIAS ===
        weekly_bullish = hma_1w_slope_aligned[i] > 0.5
        weekly_bearish = hma_1w_slope_aligned[i] < -0.5
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME ===
        is_range_market = chop_14[i] > 55
        is_trend_market = chop_14[i] < 45
        
        # === BOLLINGER BAND POSITION ===
        price_near_bb_lower = bb_position[i] < 0.15
        price_near_bb_upper = bb_position[i] > 0.85
        price_below_bb_lower = close[i] < bb_lower[i]
        price_above_bb_upper = close[i] > bb_upper[i]
        
        # === CONNORS RSI ===
        crsi_oversold = crsi[i] < 25
        crsi_overbought = crsi[i] > 75
        crsi_extreme_low = crsi[i] < 20
        crsi_extreme_high = crsi[i] > 80
        
        # === RSI FILTER ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        # Increase size when regime + trend align
        if weekly_bullish and not weekly_bearish:
            current_size = MAX_SIZE
        elif weekly_bearish and not weekly_bullish:
            current_size = MAX_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES - Simpler conditions for more trades
        long_conditions = 0
        
        # Condition 1: Range market + CRSI oversold (mean reversion)
        if is_range_market and crsi_oversold:
            long_conditions += 2
        
        # Condition 2: Price near BB lower + CRSI low
        if price_near_bb_lower and crsi[i] < 30:
            long_conditions += 2
        
        # Condition 3: Weekly bullish + pullback (trend follow)
        if weekly_bullish and crsi[i] < 40:
            long_conditions += 1
        
        # Condition 4: Price above 1w HMA + oversold (bull market dip)
        if price_above_1w_hma and crsi_extreme_low:
            long_conditions += 2
        
        # Condition 5: RSI oversold + BB lower (classic mean revert)
        if rsi_oversold and price_below_bb_lower:
            long_conditions += 2
        
        # Condition 6: Very extreme CRSI (capitulation)
        if crsi[i] < 15:
            long_conditions += 3
        
        if long_conditions >= 3:
            new_signal = current_size
        elif long_conditions >= 2 and bars_since_last_trade > 30:
            new_signal = BASE_SIZE
        
        # SHORT ENTRIES
        short_conditions = 0
        
        # Condition 1: Range market + CRSI overbought
        if is_range_market and crsi_overbought:
            short_conditions += 2
        
        # Condition 2: Price near BB upper + CRSI high
        if price_near_bb_upper and crsi[i] > 70:
            short_conditions += 2
        
        # Condition 3: Weekly bearish + rally (trend follow)
        if weekly_bearish and crsi[i] > 60:
            short_conditions += 1
        
        # Condition 4: Price below 1w HMA + overbought (bear market rally)
        if price_below_1w_hma and crsi_extreme_high:
            short_conditions += 2
        
        # Condition 5: RSI overbought + BB upper
        if rsi_overbought and price_above_bb_upper:
            short_conditions += 2
        
        # Condition 6: Very extreme CRSI
        if crsi[i] > 85:
            short_conditions += 3
        
        if short_conditions >= 3:
            new_signal = -current_size
        elif short_conditions >= 2 and bars_since_last_trade > 30:
            new_signal = -BASE_SIZE
        
        # === FREQUENCY SAFEGUARD ===
        # Force trade if no signal for 60 bars (~60 days on 1d)
        if bars_since_last_trade > 60 and new_signal == 0.0 and not in_position:
            if weekly_bullish and crsi[i] < 40:
                new_signal = BASE_SIZE * 0.6
            elif weekly_bearish and crsi[i] > 60:
                new_signal = -BASE_SIZE * 0.6
            elif crsi[i] < 25:
                new_signal = BASE_SIZE * 0.5
            elif crsi[i] > 75:
                new_signal = -BASE_SIZE * 0.5
        
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
            # Exit long if trend turns bearish strongly
            if position_side > 0 and weekly_bearish and hma_1w_slope_aligned[i] < -1.0:
                regime_reversal = True
            # Exit short if trend turns bullish strongly
            if position_side < 0 and weekly_bullish and hma_1w_slope_aligned[i] > 1.0:
                regime_reversal = True
        
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