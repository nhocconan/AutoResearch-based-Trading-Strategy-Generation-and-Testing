#!/usr/bin/env python3
"""
Experiment #103: 1d Primary + 1w HTF — Fisher Transform + Choppiness Regime

Hypothesis: Current best (12h Chop+Connors) works but Connors RSI is optimized for mean
reversion only. For 1d timeframe in bear/range markets (2025 test period), we need:
1. Ehlers Fisher Transform - catches reversals better than RSI in bear rallies
2. Choppiness Index - regime switch (range vs trend)
3. 1w HMA slope - major trend bias (very slow, reduces counter-trend trades)
4. ATR volatility filter - only trade when vol is elevated (avoid dead markets)

Why this should beat current best (Sharpe=0.220):
- Fisher Transform has superior reversal detection vs RSI (Ehlers research)
- 1d timeframe = fewer trades = less fee drag (target 15-30 trades/year)
- 1w HTF provides stronger trend filter than 1d (current best uses 1d HTF)
- Volatility filter avoids trading in low-vol chop (reduces whipsaws)

Key differences from failed strategies:
- NOT using Connors RSI (overused, diminishing returns)
- NOT using Donchian breakouts (failed in #096, #097, #102)
- Using Fisher Transform (only tried in #099, #102 but with wrong TF combo)
- 1d primary is proven to work better than 4h/12h for this market regime

Timeframe: 1d (REQUIRED for this experiment)
HTF: 1w via mtf_data.get_htf_data() — called ONCE before loop
Position sizing: 0.28 discrete (slightly conservative for 1d)
Stoploss: 2.5 * ATR(14) trailing
Target trades: 15-30/year per symbol
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_fisher_chop_hma_1w_v1"
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

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian distribution for better reversal signals.
    
    Formula:
    1. Calculate typical price: (high + low) / 2
    2. Normalize: (price - lowest) / (highest - lowest)
    3. Scale: 0.66 * normalized + 0.67 * prev_scaled
    4. Fisher: 0.5 * ln((1 + scaled) / (1 - scaled))
    
    Entry signals:
    - Long: Fisher crosses above -1.5 (oversold reversal)
    - Short: Fisher crosses below +1.5 (overbought reversal)
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    # Typical price
    typical = (high_s + low_s) / 2.0
    
    # Normalize over period
    lowest = typical.rolling(window=period, min_periods=period).min()
    highest = typical.rolling(window=period, min_periods=period).max()
    
    price_range = highest - lowest
    price_range = price_range.replace(0, 1e-10)  # avoid div by zero
    
    normalized = (typical - lowest) / price_range
    
    # Smooth with EMA-like weighting
    scaled = np.zeros(len(typical))
    scaled[0] = 0.0
    for i in range(1, len(typical)):
        scaled[i] = 0.66 * normalized.iloc[i] + 0.67 * scaled[i-1]
        scaled[i] = np.clip(scaled[i], -0.999, 0.999)  # prevent ln domain error
    
    # Fisher transform
    fisher = np.zeros(len(typical))
    for i in range(len(typical)):
        if abs(scaled[i]) < 0.999:
            fisher[i] = 0.5 * np.log((1 + scaled[i]) / (1 - scaled[i]))
        else:
            fisher[i] = fisher[i-1] if i > 0 else 0.0
    
    # Fisher trigger line (1-period lag)
    fisher_trigger = np.roll(fisher, 1)
    fisher_trigger[0] = fisher[0]
    
    return fisher, fisher_trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = range market (mean revert)
    CHOP < 38.2 = trend market (trend follow)
    """
    atr_values = calculate_atr(high, low, close, period)
    
    # Rolling sum of ATR
    atr_sum = pd.Series(atr_values).rolling(window=period, min_periods=period).sum().values
    
    # Highest High and Lowest Low over period
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Choppiness calculation
    price_range = highest_high - lowest_low
    price_range = np.where(price_range == 0, 1e-10, price_range)  # avoid div by zero
    
    chop = 100 * np.log10(atr_sum / price_range) / np.log10(period)
    chop = np.clip(chop, 0, 100)
    
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
    """Calculate HMA slope over lookback period (percentage change)."""
    slope = np.zeros(len(hma_values))
    for i in range(lookback, len(hma_values)):
        if hma_values[i - lookback] != 0 and not np.isnan(hma_values[i - lookback]):
            slope[i] = (hma_values[i] - hma_values[i - lookback]) / abs(hma_values[i - lookback]) * 100
        else:
            slope[i] = 0.0
    return slope

def calculate_vol_ratio(atr_short=7, atr_long=30, high=None, low=None, close=None):
    """Calculate ATR ratio for volatility spike detection."""
    atr_s = calculate_atr(high, low, close, atr_short)
    atr_l = calculate_atr(high, low, close, atr_long)
    
    vol_ratio = np.zeros(len(close))
    for i in range(len(close)):
        if atr_l[i] > 0 and not np.isnan(atr_l[i]):
            vol_ratio[i] = atr_s[i] / atr_l[i]
        else:
            vol_ratio[i] = 1.0
    
    return vol_ratio

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
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_1w_21_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_21)
    hma_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_slope)
    
    # Calculate 1d indicators
    atr_14 = calculate_atr(high, low, close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    fisher, fisher_trigger = calculate_fisher_transform(high, low, 9)
    vol_ratio = calculate_vol_ratio(7, 30, high, low, close)
    
    # Additional trend confirmation
    hma_21 = calculate_hma(close, 21)
    hma_50 = calculate_hma(close, 50)
    rsi_14 = calculate_rsi(close, 14)
    
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_1w_21_aligned[i]) or np.isnan(hma_1w_slope_aligned[i]):
            continue
        
        if np.isnan(chop_14[i]) or np.isnan(fisher[i]):
            continue
        
        if np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            continue
        
        # === 1W TREND BIAS (MAJOR) ===
        # HMA slope > 1.0 = strong bullish bias
        # HMA slope < -1.0 = strong bearish bias
        # Between = neutral (allow both directions with weaker signals)
        trend_1w_bullish = hma_1w_slope_aligned[i] > 1.0
        trend_1w_bearish = hma_1w_slope_aligned[i] < -1.0
        trend_1w_neutral = not trend_1w_bullish and not trend_1w_bearish
        
        # Price vs 1w HMA for additional confirmation
        price_above_1w_hma = close[i] > hma_1w_21_aligned[i]
        price_below_1w_hma = close[i] < hma_1w_21_aligned[i]
        
        # === CHOPPINESS REGIME DETECTION ===
        # CHOP > 61.8 = range market (mean revert strategy)
        # CHOP < 38.2 = trend market (trend follow strategy)
        # Use slightly relaxed thresholds for more trades
        is_range_market = chop_14[i] > 58
        is_trend_market = chop_14[i] < 42
        
        # === VOLATILITY FILTER ===
        # Only trade when vol_ratio > 1.3 (elevated volatility)
        # Avoid dead markets where signals whipsaw
        vol_elevated = vol_ratio[i] > 1.2
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 (oversold reversal)
        # Short: Fisher crosses below +1.5 (overbought reversal)
        fisher_cross_long = (fisher[i] > -1.5) and (fisher_trigger[i] <= -1.5)
        fisher_cross_short = (fisher[i] < 1.5) and (fisher_trigger[i] >= 1.5)
        
        # Also allow extreme Fisher values without cross (stronger signals)
        fisher_extreme_long = fisher[i] < -2.0
        fisher_extreme_short = fisher[i] > 2.0
        
        # === 1D TREND CONFIRMATION ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # RSI confirmation
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === POSITION SIZING ===
        current_size = BASE_SIZE
        
        # Reduce size in low volatility
        if not vol_elevated:
            current_size = BASE_SIZE * 0.5
        
        # Reduce size in transitional chop regime
        if not is_range_market and not is_trend_market:
            current_size = current_size * 0.7
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # LONG ENTRIES
        long_conditions = []
        
        if is_range_market:
            # Mean reversion in range: Fisher extreme + RSI oversold
            if (fisher_extreme_long or fisher_cross_long) and rsi_oversold:
                if trend_1w_bullish or price_above_1w_hma or trend_1w_neutral:
                    long_conditions.append(True)
        elif is_trend_market:
            # Trend following: Fisher pullback + 1w bullish
            if trend_1w_bullish and (fisher_cross_long or fisher[i] < -1.0):
                if hma_bullish or price_above_1w_hma:
                    long_conditions.append(True)
            # Also allow if 1d HMA bullish with Fisher reversal
            elif hma_bullish and fisher_cross_long and vol_elevated:
                long_conditions.append(True)
        else:
            # Transitional: need strong Fisher + RSI confluence
            if fisher_extreme_long and rsi_oversold and vol_elevated:
                long_conditions.append(True)
        
        # SHORT ENTRIES
        short_conditions = []
        
        if is_range_market:
            # Mean reversion in range: Fisher extreme + RSI overbought
            if (fisher_extreme_short or fisher_cross_short) and rsi_overbought:
                if trend_1w_bearish or price_below_1w_hma or trend_1w_neutral:
                    short_conditions.append(True)
        elif is_trend_market:
            # Trend following: Fisher pullback + 1w bearish
            if trend_1w_bearish and (fisher_cross_short or fisher[i] > 1.0):
                if hma_bearish or price_below_1w_hma:
                    short_conditions.append(True)
            # Also allow if 1d HMA bearish with Fisher reversal
            elif hma_bearish and fisher_cross_short and vol_elevated:
                short_conditions.append(True)
        else:
            # Transitional: need strong Fisher + RSI confluence
            if fisher_extreme_short and rsi_overbought and vol_elevated:
                short_conditions.append(True)
        
        # Apply long/short signals
        if len(long_conditions) > 0:
            new_signal = current_size
        elif len(short_conditions) > 0:
            new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 90 bars (~90 days on 1d), allow weaker entry
        if bars_since_last_trade > 90 and new_signal == 0.0 and not in_position:
            if trend_1w_bullish and fisher[i] < -1.0 and rsi_14[i] < 40:
                new_signal = current_size * 0.5
            elif trend_1w_bearish and fisher[i] > 1.0 and rsi_14[i] > 60:
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
        
        # === REGIME REVERSAL EXIT ===
        # Exit if 1w trend strongly reverses against position
        regime_reversal = False
        if in_position and position_side != 0:
            # Exit long if 1w becomes strongly bearish
            if position_side > 0 and trend_1w_bearish and hma_1w_slope_aligned[i] < -2.0:
                regime_reversal = True
            # Exit short if 1w becomes strongly bullish
            if position_side < 0 and trend_1w_bullish and hma_1w_slope_aligned[i] > 2.0:
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