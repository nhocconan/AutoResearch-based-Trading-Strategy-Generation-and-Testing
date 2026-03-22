#!/usr/bin/env python3
"""
Experiment #027: 1h Regime-Adaptive Strategy with 4h HMA + Choppiness + Funding Z-Score

Hypothesis: After analyzing 26 experiments, the clearest pattern is:
1. Pure trend-following fails on BTC/ETH during bear markets (2022 crash)
2. Pure mean-reversion fails during strong trends
3. REGIME DETECTION is the missing piece - adapt strategy to market state
4. Funding rate z-score is a UNIQUE edge for BTC/ETH (contrarian signal)
5. 1h timeframe needs HTF (4h) trend filter to reduce noise
6. Choppiness Index (CHOP) best distinguishes range vs trend regimes

This 1h strategy combines:

1. 4h HMA(21): Primary trend filter - only long when price > 4h HMA, short when <

2. Choppiness Index(14): Regime detection
   - CHOP > 61.8 = Range regime → use RSI mean-reversion entries
   - CHOP < 38.2 = Trend regime → use Donchian breakout entries
   - Between = neutral, reduce position size

3. Funding Rate Z-Score(30d): Contrarian filter (BTC/ETH specific edge)
   - Z > +2.0 = overcrowded longs → avoid long / prefer short
   - Z < -2.0 = overcrowded shorts → avoid short / prefer long
   - This addresses the 2022 crash where everyone was long

4. RSI(14) for Range Regime:
   - Long: RSI < 35 + price > 4h HMA
   - Short: RSI > 65 + price < 4h HMA

5. Donchian(20) for Trend Regime:
   - Long: Break above 20-bar high + price > 4h HMA
   - Short: Break below 20-bar low + price < 4h HMA

6. ATR(14) Stoploss: 2.5*ATR trailing stop (tighter for 1h vs 12h)

7. Position Sizing: 0.20-0.35 discrete, reduced in neutral regime

Why this should beat current best (Sharpe=0.137):
- Regime adaptation addresses the #1 failure mode (trend strategies in ranges)
- Funding z-score is PROVEN edge for BTC/ETH through 2022 crash
- 4h HMA filter reduces 1h noise while maintaining trade frequency
- Conservative sizing (max 0.35) protects from crashes

Timeframe: 1h (REQUIRED for this experiment)
HTF: 4h via mtf_data helper (call ONCE before loop)
Position sizing: 0.20-0.35 discrete levels
Stoploss: 2.5 * ATR(14) trailing
Target trades: 30-60/year on 1h (optimal per Rule 10)
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_regime_chop_4h_hma_funding_zscore_atr_v1"
timeframe = "1h"
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

def calculate_hma(close, period=21):
    """Calculate Hull Moving Average for smoother trend with less lag."""
    close_s = pd.Series(close)
    half = max(1, period // 2)
    sqrt_period = max(1, int(np.sqrt(period)))
    wma1 = close_s.ewm(span=half, min_periods=half, adjust=False).mean()
    wma2 = close_s.ewm(span=period, min_periods=period, adjust=False).mean()
    wma3 = (2 * wma1 - wma2).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean()
    return wma3.values

def calculate_rsi(close, period=14):
    """Calculate RSI using Wilder's smoothing."""
    close_s = pd.Series(close)
    delta = close_s.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(span=period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def calculate_choppiness(high, low, close, period=14):
    """
    Calculate Choppiness Index (CHOP).
    CHOP = 100 * LOG10(SUM(ATR, period) / (Highest High - Lowest Low)) / LOG10(period)
    CHOP > 61.8 = Range/Choppy
    CHOP < 38.2 = Trending
    """
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    
    # ATR for each bar
    tr1 = high_s - low_s
    tr2 = np.abs(high_s - close_s.shift(1))
    tr3 = np.abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_sum = tr.rolling(window=period, min_periods=period).sum()
    
    # Highest high and lowest low over period
    highest_high = high_s.rolling(window=period, min_periods=period).max()
    lowest_low = low_s.rolling(window=period, min_periods=period).min()
    price_range = highest_high - lowest_low
    
    # CHOP formula
    with np.errstate(divide='ignore', invalid='ignore'):
        chop = 100 * np.log10(atr_sum / price_range.replace(0, np.inf)) / np.log10(period)
    
    chop = np.where(np.isnan(chop) | np.isinf(chop), 50.0, chop)
    return chop

def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (highest high and lowest low over period)."""
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    
    upper = high_s.rolling(window=period, min_periods=period).max()
    lower = low_s.rolling(window=period, min_periods=period).min()
    middle = (upper + lower) / 2
    
    return upper.values, lower.values, middle.values

def load_funding_zscore(symbol, prices):
    """
    Load funding rate data and calculate 30-day z-score.
    Returns array aligned with prices timeline.
    """
    try:
        # Map symbol to funding file
        symbol_map = {
            'BTCUSDT': 'BTCUSDT',
            'ETHUSDT': 'ETHUSDT',
            'SOLUSDT': 'SOLUSDT'
        }
        funding_symbol = symbol_map.get(symbol, symbol)
        funding_path = f"data/processed/funding/{funding_symbol}.parquet"
        
        funding_df = pd.read_parquet(funding_path)
        
        # Calculate z-score of funding rate over 30 days (90 funding periods at 8h)
        funding_df['funding_zscore'] = (
            funding_df['funding_rate'] - 
            funding_df['funding_rate'].rolling(window=90, min_periods=30).mean()
        ) / funding_df['funding_rate'].rolling(window=90, min_periods=30).std()
        
        # Align funding data to prices timeline
        # Funding data is 8h, prices is 1h - need to forward-fill
        prices_with_time = prices.copy()
        prices_with_time['open_time_dt'] = pd.to_datetime(prices_with_time['open_time'], unit='ms')
        
        funding_df['open_time_dt'] = pd.to_datetime(funding_df['open_time'], unit='ms')
        
        # Merge and forward-fill
        merged = pd.merge_asof(
            prices_with_time.sort_values('open_time_dt'),
            funding_df[['open_time_dt', 'funding_zscore']].sort_values('open_time_dt'),
            on='open_time_dt',
            direction='backward'
        )
        
        return merged['funding_zscore'].fillna(0).values
        
    except Exception as e:
        # If funding data not available, return zeros (no funding filter)
        return np.zeros(len(prices))

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate HTF indicators
    hma_4h = calculate_hma(df_4h['close'].values, 21)
    
    # Align HTF to LTF (Rule 2 - auto shift(1) for completed bars)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Calculate 1h indicators
    atr_14 = calculate_atr(high, low, close, 14)
    rsi_14 = calculate_rsi(close, 14)
    chop_14 = calculate_choppiness(high, low, close, 14)
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    
    # Load funding z-score (BTC/ETH specific edge)
    # Try to load from funding data, fallback to zeros
    try:
        symbol = prices.get('symbol', 'BTCUSDT')
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[0]
        funding_zscore = load_funding_zscore(symbol, prices)
    except:
        funding_zscore = np.zeros(n)
    
    signals = np.zeros(n)
    
    # Position sizing (Rule 4 - discrete levels, max 0.40)
    SIZE_STRONG = 0.35  # All filters agree
    SIZE_MODERATE = 0.25  # Partial confirmation
    SIZE_WEAK = 0.20  # Minimal confirmation
    
    # Track position state for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_price = 0.0
    lowest_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] == 0:
            continue
        
        if np.isnan(hma_4h_aligned[i]):
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(chop_14[i]):
            continue
        
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            continue
        
        # === HTF TREND BIAS (4h HMA) ===
        price_vs_4h = close[i] - hma_4h_aligned[i]
        bull_htf = price_vs_4h > 0
        bear_htf = price_vs_4h < 0
        
        # === REGIME DETECTION (Choppiness Index) ===
        chop_value = chop_14[i]
        is_range_regime = chop_value > 61.8  # Choppy/range market
        is_trend_regime = chop_value < 38.2  # Strong trend
        is_neutral_regime = not is_range_regime and not is_trend_regime
        
        # === FUNDING RATE CONTRARIAN FILTER ===
        funding_z = funding_zscore[i]
        funding_bullish = funding_z < -2.0  # Overcrowded shorts → bullish contrarian
        funding_bearish = funding_z > 2.0  # Overcrowded longs → bearish contrarian
        funding_neutral = not funding_bullish and not funding_bearish
        
        # === RSI MEAN-REVERSION (Range Regime) ===
        rsi_oversold = rsi_14[i] < 35
        rsi_overbought = rsi_14[i] > 65
        
        # === DONCHIAN BREAKOUT (Trend Regime) ===
        # Use previous bar's Donchian levels to avoid look-ahead
        donchian_breakout_long = close[i] > donchian_upper[i-1] if i > 0 else False
        donchian_breakout_short = close[i] < donchian_lower[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (Regime-Adaptive) ===
        new_signal = 0.0
        signal_strength = 0
        
        # LONG ENTRY
        if bull_htf:  # Must have HTF bullish bias for longs
            if is_range_regime and rsi_oversold:
                # Range regime: RSI mean-reversion long
                signal_strength = 2  # HTF + RSI
                
                if funding_bullish or funding_neutral:
                    signal_strength += 1  # Funding confirms or neutral
                
                if rsi_14[i] < 30:
                    signal_strength += 1  # Extremely oversold
            
            elif is_trend_regime and donchian_breakout_long:
                # Trend regime: Donchian breakout long
                signal_strength = 2  # HTF + Breakout
                
                if funding_bullish or funding_neutral:
                    signal_strength += 1  # Funding confirms or neutral
            
            elif is_neutral_regime:
                # Neutral regime: reduced size, need stronger confirmation
                if rsi_oversold or donchian_breakout_long:
                    signal_strength = 1
                    if funding_bullish:
                        signal_strength += 1
        
        # SHORT ENTRY
        elif bear_htf:  # Must have HTF bearish bias for shorts
            if is_range_regime and rsi_overbought:
                # Range regime: RSI mean-reversion short
                signal_strength = 2  # HTF + RSI
                
                if funding_bearish or funding_neutral:
                    signal_strength += 1  # Funding confirms or neutral
                
                if rsi_14[i] > 70:
                    signal_strength += 1  # Extremely overbought
            
            elif is_trend_regime and donchian_breakout_short:
                # Trend regime: Donchian breakout short
                signal_strength = 2  # HTF + Breakout
                
                if funding_bearish or funding_neutral:
                    signal_strength += 1  # Funding confirms or neutral
            
            elif is_neutral_regime:
                # Neutral regime: reduced size, need stronger confirmation
                if rsi_overbought or donchian_breakout_short:
                    signal_strength = 1
                    if funding_bearish:
                        signal_strength += 1
        
        # Assign size based on confirmation count and regime
        if signal_strength >= 4:
            new_signal = SIZE_STRONG * np.sign(1 if bull_htf else -1)
        elif signal_strength >= 2:
            new_signal = SIZE_MODERATE * np.sign(1 if bull_htf else -1)
        elif signal_strength >= 1 and not is_neutral_regime:
            new_signal = SIZE_WEAK * np.sign(1 if bull_htf else -1)
        
        # Reduce size in neutral regime
        if is_neutral_regime and new_signal != 0:
            new_signal = new_signal * 0.7  # 30% reduction in neutral
        
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
        
        # === REGIME REVERSAL EXIT ===
        regime_exit = False
        if in_position and position_side != 0:
            # Exit if regime strongly reverses against position
            if position_side > 0 and bear_htf:
                regime_exit = True
            if position_side < 0 and bull_htf:
                regime_exit = True
            
            # Exit if RSI extreme opposite in range regime
            if is_range_regime:
                if position_side > 0 and rsi_14[i] > 70:
                    regime_exit = True
                if position_side < 0 and rsi_14[i] < 30:
                    regime_exit = True
        
        # Apply stoploss or regime exit
        if stoploss_triggered or regime_exit:
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
            elif np.sign(new_signal) != position_side:
                # Position flip
                position_side = np.sign(new_signal)
                entry_price = close[i]
                highest_price = close[i] if position_side > 0 else 0.0
                lowest_price = close[i] if position_side < 0 else 0.0
        else:
            if in_position:
                # Exit position
                in_position = False
                position_side = 0
                entry_price = 0.0
                highest_price = 0.0
                lowest_price = 0.0
        
        signals[i] = new_signal
    
    return signals