#!/usr/bin/env python3
"""
Experiment #024: 4h KAMA + Volatility Regime + 12h/1d HMA Trend Filter

Hypothesis: 4h primary timeframe with adaptive trend following (KAMA) combined with
volatility regime detection and HTF trend filter will work better than pure mean
reversion or pure trend strategies. Key innovations:

1. KAMA(10) with Efficiency Ratio - adapts to market noise, reduces whipsaw
2. Volatility regime detection: ATR(7)/ATR(30) ratio determines entry type
   - High vol (>1.8): mean reversion at Bollinger extremes
   - Low vol (<1.2): trend following on KAMA breakout
3. Choppiness Index(14) regime filter: CHOP>61.8 = range, CHOP<38.2 = trend
4. 12h HMA(21) + 1d HMA(21) dual HTF bias - both must agree for strong signal
5. ADX(14) confirmation for trend entries (ADX>25)
6. Asymmetric exits: trail stop in trends, fixed target in mean reversion
7. Discrete sizing: 0.25 base, 0.30 high conviction, 0.20 low conviction

Why this should work:
- KAMA adapts to volatility, unlike static EMA/SMA
- Vol regime switching captures both panic reversals and clean trends
- Dual HTF filter prevents counter-trend trades (major failure mode in 2022/2025)
- 4h TF targets 20-50 trades/year (optimal fee efficiency)
- Choppiness filter avoids trading in unclear regimes

Timeframe: 4h (REQUIRED for this experiment)
HTF: 12h and 1d via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.30 discrete
Stoploss: 2.5 * ATR(14) trailing
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_vol_regime_12h_1d_hma_v1"
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
    """Calculate RSI using Wilder's smoothing."""
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    loss_avg = np.where(loss_avg == 0, 1e-10, loss_avg)
    rs = gain_avg / loss_avg
    rsi = 100 - (100 / (1 + rs))
    rsi = np.clip(rsi, 0, 100)
    return rsi

def calculate_kama(close, er_period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average.
    Adapts smoothing based on market efficiency (trend vs noise).
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    change = np.abs(close - np.roll(close, er_period))
    change[:er_period] = np.nan
    
    volatility = np.zeros(n)
    for i in range(er_period, n):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
    
    volatility[:er_period] = np.nan
    volatility = np.where(volatility == 0, 1e-10, volatility)
    
    er = change / volatility
    er = np.nan_to_num(er, nan=0.0)
    er = np.clip(er, 0, 1)
    
    # Calculate smoothing constants
    fast_sc = (2.0 / (fast_period + 1)) ** 2
    slow_sc = (2.0 / (slow_period + 1)) ** 2
    
    # KAMA calculation
    kama[er_period] = close[er_period]
    for i in range(er_period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        elif minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr == 0, 1e-10, atr)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, min_periods=period, adjust=False).mean().values / np.where(atr == 0, 1e-10, atr)
    
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) == 0, 1e-10, (plus_di + minus_di))
    adx = pd.Series(dx).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    return adx, plus_di, minus_di

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index.
    CHOP > 61.8 = ranging market
    CHOP < 38.2 = trending market
    """
    n = len(close)
    chop = np.zeros(n)
    
    for i in range(period, n):
        highest_high = np.max(high[i-period:i+1])
        lowest_low = np.min(low[i-period:i+1])
        
        if highest_high == lowest_low:
            chop[i] = 100
        else:
            atr_sum = 0
            for j in range(i-period+1, i+1):
                tr = max(high[j] - low[j], 
                        abs(high[j] - close[j-1]), 
                        abs(low[j] - close[j-1]))
                atr_sum += tr
            
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
    
    chop = np.clip(chop, 0, 100)
    return chop

def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands."""
    close_s = pd.Series(close)
    middle = close_s.rolling(window=period, min_periods=period).mean().values
    std = close_s.rolling(window=period, min_periods=period).std().values
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower

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
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 12h indicators
    hma_12h_21 = calculate_hma(df_12h['close'].values, 21)
    hma_12h_21_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_21)
    
    # Calculate 1d indicators
    hma_1d_21 = calculate_hma(df_1d['close'].values, 21)
    hma_1d_21_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_21)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    atr_7 = calculate_atr(high, low, close, 7)
    atr_30 = calculate_atr(high, low, close, 30)
    kama_10 = calculate_kama(close, er_period=10, fast_period=2, slow_period=30)
    adx, plus_di, minus_di = calculate_adx(high, low, close, 14)
    choppiness = calculate_choppiness(high, low, close, 14)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2.0)
    rsi_14 = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    BASE_SIZE = 0.25
    HIGH_CONV_SIZE = 0.30
    LOW_CONV_SIZE = 0.20
    
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
        
        if np.isnan(hma_12h_21_aligned[i]) or np.isnan(hma_1d_21_aligned[i]):
            continue
        
        if np.isnan(kama_10[i]) or np.isnan(adx[i]) or np.isnan(choppiness[i]):
            continue
        
        # === HTF TREND BIAS (12h + 1d) ===
        # Both HTFs must agree for strong bias
        htf_bullish = (close[i] > hma_12h_21_aligned[i]) and (close[i] > hma_1d_21_aligned[i])
        htf_bearish = (close[i] < hma_12h_21_aligned[i]) and (close[i] < hma_1d_21_aligned[i])
        htf_neutral = not htf_bullish and not htf_bearish
        
        # === VOLATILITY REGIME ===
        # ATR ratio determines market state
        atr_ratio = atr_7[i] / atr_30[i] if atr_30[i] > 0 else 1.0
        high_vol_regime = atr_ratio > 1.8  # Vol spike - mean reversion likely
        low_vol_regime = atr_ratio < 1.2   # Calm - trend following likely
        normal_vol = not high_vol_regime and not low_vol_regime
        
        # === CHOPPINESS REGIME ===
        chop_range = choppiness[i] > 61.8  # Ranging market
        chop_trend = choppiness[i] < 38.2  # Trending market
        chop_neutral = not chop_range and not chop_trend
        
        # === ADX TREND STRENGTH ===
        strong_trend = adx[i] > 25
        weak_trend = adx[i] < 20
        
        # === POSITION SIZING ===
        # Higher conviction when HTF + LTF agree + strong trend
        conviction = 0
        if htf_bullish or htf_bearish:
            conviction += 1
        if strong_trend:
            conviction += 1
        if not chop_range:
            conviction += 1
        
        if conviction >= 2:
            current_size = HIGH_CONV_SIZE
        elif conviction == 1:
            current_size = BASE_SIZE
        else:
            current_size = LOW_CONV_SIZE
        
        # === ENTRY LOGIC ===
        new_signal = 0.0
        bars_since_last_trade = i - last_trade_bar
        
        # REGIME 1: High Volatility + Range (Mean Reversion)
        # Enter when price at BB extreme + RSI extreme + HTF bias
        if high_vol_regime and chop_range:
            # Long: price at lower BB + RSI oversold + HTF not bearish
            if close[i] < bb_lower[i] and rsi_14[i] < 30 and not htf_bearish:
                new_signal = current_size
            # Short: price at upper BB + RSI overbought + HTF not bullish
            elif close[i] > bb_upper[i] and rsi_14[i] > 70 and not htf_bullish:
                new_signal = -current_size
        
        # REGIME 2: Low Volatility + Trend (Trend Following)
        # Enter on KAMA breakout + ADX confirmation + HTF agreement
        elif low_vol_regime and chop_trend and strong_trend:
            # Long: price above KAMA + ADX rising + HTF bullish
            if close[i] > kama_10[i] and plus_di[i] > minus_di[i] and htf_bullish:
                new_signal = current_size
            # Short: price below KAMA + ADX rising + HTF bearish
            elif close[i] < kama_10[i] and minus_di[i] > plus_di[i] and htf_bearish:
                new_signal = -current_size
        
        # REGIME 3: Normal Volatility (Hybrid)
        # Use KAMA direction + RSI pullback entry
        elif normal_vol:
            # Long: HTF bullish + price pullback to KAMA + RSI not overbought
            if htf_bullish and close[i] > kama_10[i] and rsi_14[i] < 60 and rsi_14[i] > 40:
                new_signal = current_size
            # Short: HTF bearish + price rally to KAMA + RSI not oversold
            elif htf_bearish and close[i] < kama_10[i] and rsi_14[i] > 40 and rsi_14[i] < 60:
                new_signal = -current_size
        
        # === FREQUENCY SAFEGUARD ===
        # If no trades for 72 bars (~12 days on 4h), allow weaker entry
        if bars_since_last_trade > 72 and new_signal == 0.0 and not in_position:
            if htf_bullish and rsi_14[i] < 35:
                new_signal = current_size * 0.8
            elif htf_bearish and rsi_14[i] > 65:
                new_signal = -current_size * 0.8
        
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
        
        # === REGIME CHANGE EXIT ===
        # Exit if market regime changes against position
        regime_exit = False
        if in_position and position_side != 0:
            # Long position: exit if high vol + range regime starts
            if position_side > 0 and high_vol_regime and chop_range and rsi_14[i] > 60:
                regime_exit = True
            # Short position: exit if high vol + range regime starts
            if position_side < 0 and high_vol_regime and chop_range and rsi_14[i] < 40:
                regime_exit = True
        
        # === HTF TREND REVERSAL EXIT ===
        htf_reversal = False
        if in_position and position_side != 0:
            if position_side > 0 and htf_bearish:
                htf_reversal = True
            if position_side < 0 and htf_bullish:
                htf_reversal = True
        
        # === ADX WEAKNESS EXIT ===
        adx_weakness = False
        if in_position and position_side != 0 and strong_trend:
            # If ADX drops below 20, trend is weakening
            if adx[i] < 20:
                adx_weakness = True
        
        # Apply stoploss or exits
        if stoploss_triggered or regime_exit or htf_reversal or adx_weakness:
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
                # Reversal - close old position, open new
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
                last_trade_bar = i
            # If same side, maintain position (no update needed)
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